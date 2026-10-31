"""
Stage 1 - CLEANING (pandas)

Takes a "raw" DataFrame (every column read as text) and returns a tidy DataFrame
plus a CleaningReport describing exactly what was changed.

Works on any CSV - nothing here is hard-coded to one dataset. The rules are
driven by column *content* and a few column-name hints (price, qty, date ...).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from difflib import get_close_matches

import numpy as np
import pandas as pd

# Values that really mean "missing"
NULL_TOKENS = {
    "", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?",
    "missing", "unknown", "undefined", "not available",
}

DATE_NAME_HINT = re.compile(r"date|time|timestamp|_dt$|^dt_|created|updated|\bday\b", re.I)
DATE_VALUE_HINT = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"            # 2025-01-31, 31/01/2025
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"         # 31 Jan 2025
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}"       # January 31, 2025
    r"|\d{1,2}-[A-Za-z]{3,9}-\d{2,4}"             # 31-Jan-2025
)
ID_LIKE = re.compile(r"(^|_)(id|uuid|key|index|sku|zip|pin|phone|mobile|code)$", re.I)
NON_NEGATIVE_HINT = re.compile(
    r"qty|quantity|units|price|amount|revenue|sales|cost|age|count|total", re.I
)


@dataclass(frozen=True)
class CleaningOptions:
    dayfirst: bool = True            # 01/02/2025 -> 1 Feb (set False for US-style dates)
    fuzzy_merge: bool = True         # merge near-duplicate labels such as "Electronis"
    drop_duplicates: bool = True
    negatives_invalid: bool = True   # negative qty / price / amount -> missing
    outliers: str = "cap"            # "cap" | "remove" | "keep"
    outlier_factor: float = 3.0      # distance from the quartiles, in IQRs
    numeric_missing: str = "median"  # "median" | "mean" | "zero" | "drop" | "keep"
    text_missing: str = "Unknown"    # label for missing text ("" = leave empty)


@dataclass
class CleaningReport:
    rows_before: int = 0
    rows_after: int = 0
    steps: list = field(default_factory=list)
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)

    def log(self, step: str, detail: str, affected: int = 0) -> None:
        self.steps.append({"Step": step, "What happened": detail, "Affected": int(affected)})

    @property
    def total_fixes(self) -> int:
        return sum(s["Affected"] for s in self.steps)

    def steps_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["Step", "What happened", "Affected"])


# ----------------------------------------------------------------------------- helpers
def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return name or "column"


def _to_number(s: pd.Series) -> pd.Series:
    """'Rs. 1,200.50' / '$1,200' / '12%' / '(45)' -> float."""
    t = s.astype(object).astype(str)
    t = t.str.replace(r"(?i)rs\.?|inr|usd|eur", "", regex=True)
    t = t.str.replace(r"[\$€£₹,%\s]", "", regex=True)
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)      # accounting negatives
    return pd.to_numeric(t, errors="coerce")


def _to_datetime(s: pd.Series, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):                         # older pandas
            out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return out


def _missing_counts(df: pd.DataFrame) -> dict:
    return {c: int(df[c].isna().sum()) for c in df.columns}


# ----------------------------------------------------------------------------- steps
def _standardize_columns(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    new, seen = [], {}
    for c in df.columns:
        base = _snake(c)
        seen[base] = seen.get(base, 0) + 1
        new.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    changed = sum(a != b for a, b in zip(df.columns, new))
    df = df.copy()
    df.columns = new
    if changed:
        rep.log("Column names", "Converted headers to clean snake_case (e.g. ' Order Date ' -> 'order_date')", changed)
    return df


def _trim_and_nullify(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = df.copy()
    trimmed = placeholders = 0
    for c in df.columns:
        if not _is_text(df[c]):
            continue
        original = df[c].astype(object)
        s = original.map(lambda v: re.sub(r"\s+", " ", v.strip()) if isinstance(v, str) else v)
        trimmed += int(((s != original) & original.notna()).sum())
        is_null = s.map(lambda v: isinstance(v, str) and v.lower() in NULL_TOKENS)
        placeholders += int(is_null.sum())
        df[c] = s.mask(is_null, np.nan)
    if trimmed:
        rep.log("Whitespace", "Trimmed leading/trailing/double spaces in text cells", trimmed)
    if placeholders:
        rep.log("Missing placeholders", "Turned 'N/A', 'null', '-', '?', blanks ... into real missing values", placeholders)
    return df


def _drop_empty(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    n = len(df)
    df = df.dropna(how="all")
    if n - len(df):
        rep.log("Empty rows", "Dropped rows that were completely empty", n - len(df))
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        rep.log("Empty columns", f"Dropped columns with no data: {', '.join(empty_cols)}", len(empty_cols))
    return df


def _infer_types(df: pd.DataFrame, opts: CleaningOptions, rep: CleaningReport) -> pd.DataFrame:
    df = df.copy()
    for c in list(df.columns):
        s = df[c]
        if not _is_text(s):
            continue
        present = s.dropna().astype(str)
        if present.empty:
            continue

        # ---- numbers (skip things like zip codes with leading zeros)
        if not present.str.match(r"^0\d+$").any():
            parsed = _to_number(s)
            ok = int(parsed.notna().sum())
            if ok / len(present) >= 0.8:
                lost = len(present) - ok
                df[c] = parsed
                rep.log("Data types", f"'{c}' -> number (currency symbols / commas removed where present)", ok)
                if lost:
                    rep.log("Invalid values", f"'{c}': {lost} value(s) that were not numbers became missing", lost)
                continue

        # ---- dates
        hinted = bool(DATE_NAME_HINT.search(c))
        looks_like_date = present.str.contains(DATE_VALUE_HINT).mean() >= 0.9
        if hinted or looks_like_date:
            parsed = _to_datetime(s, opts.dayfirst)
            ok = int(parsed.notna().sum())
            if ok / len(present) >= (0.6 if hinted else 0.9):
                lost = len(present) - ok
                df[c] = parsed
                rep.log("Data types", f"'{c}' -> date (handled mixed formats)", ok)
                if lost:
                    rep.log("Invalid values", f"'{c}': {lost} unreadable date(s) became missing", lost)
    return df


def _standardize_text(df: pd.DataFrame, opts: CleaningOptions, rep: CleaningReport) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if not _is_text(s) or ID_LIKE.search(c):
            continue
        s = s.astype(object)
        present = s.dropna().astype(str)
        n_unique = present.nunique()
        if n_unique == 0 or n_unique == len(present) or n_unique > 60 or present.str.contains("@").any():
            continue                                    # looks like free text / ids / emails

        key = s.map(lambda v: v.lower() if isinstance(v, str) else v)
        # most common spelling for each lower-case key
        canon = {}
        for k, grp in pd.DataFrame({"key": key, "raw": s}).dropna().groupby("key")["raw"]:
            best = grp.value_counts().index[0]
            if best.islower() or (best.isupper() and len(best) > 4):
                best = best.title()
            canon[k] = best
        case_fixed = int((s.map(lambda v: canon.get(v.lower()) if isinstance(v, str) else v) != s).where(s.notna(), False).sum())
        mapped = key.map(canon)

        fuzzy_fixed = 0
        if opts.fuzzy_merge and 2 <= len(canon) <= 30:
            freq = mapped.value_counts()
            keepers: list[str] = []
            remap = {}
            for label in freq.index:                    # most frequent first
                if re.search(r"\d", label):
                    keepers.append(label)
                    continue
                hit = get_close_matches(label.lower(), [k.lower() for k in keepers], n=1, cutoff=0.86)
                if hit:
                    remap[label] = next(k for k in keepers if k.lower() == hit[0])
                else:
                    keepers.append(label)
            if remap:
                fuzzy_fixed = int(mapped.isin(remap.keys()).sum())
                mapped = mapped.replace(remap)
                rep.log("Typos", f"'{c}': merged {', '.join(f'{a!r}->{b!r}' for a, b in list(remap.items())[:4])}", fuzzy_fixed)

        if case_fixed:
            rep.log("Text consistency", f"'{c}': unified spelling/case of repeated labels (north / NORTH / North)", case_fixed)
        df[c] = mapped
    return df


def _drop_duplicates(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates()
    if n - len(df):
        rep.log("Duplicates", "Removed exact duplicate rows (after standardising text and dates)", n - len(df))
    return df


def _invalid_negatives(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and NON_NEGATIVE_HINT.search(c) and not ID_LIKE.search(c):
            bad = df[c] < 0
            if bad.any():
                df.loc[bad, c] = np.nan
                rep.log("Impossible values", f"'{c}': negative values are not valid -> missing", int(bad.sum()))
    return df


def _derive_missing_total(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """If total = quantity x price and total is missing, recompute it."""
    def find(pattern):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and re.search(pattern, c, re.I):
                return c
    qty, price, total = find(r"qty|quantity|units"), find(r"price"), find(r"revenue|total|sales|amount")
    if qty and price and total and len({qty, price, total}) == 3:
        fill = df[total].isna() & df[qty].notna() & df[price].notna()
        if fill.any():
            df.loc[fill, total] = df.loc[fill, qty] * df.loc[fill, price]
            rep.log("Derived values", f"'{total}' rebuilt as {qty} x {price} where it was missing", int(fill.sum()))
    return df


def _outliers(df: pd.DataFrame, opts: CleaningOptions, rep: CleaningReport) -> pd.DataFrame:
    if opts.outliers == "keep":
        return df
    df = df.copy()
    drop_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or ID_LIKE.search(c) or df[c].nunique() < 8:
            continue
        s = df[c]
        # Right-skewed positive data (prices, revenue ...) is judged on a log scale so that
        # genuinely expensive-but-real values are not mistaken for errors.
        use_log = s.min() > 0 and s.skew() > 1
        x = np.log10(s) if use_log else s
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        factor = opts.outlier_factor / 2 if use_log else opts.outlier_factor   # log scale: standard 1.5 x IQR
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        if use_log:
            lo, hi = 10 ** lo, 10 ** hi
        if s.min() >= 0:
            lo = max(lo, 0.0)                       # never invent negative bounds
        if (s.dropna() % 1 == 0).all():             # whole-number column (e.g. quantity) -> whole-number limits
            lo, hi = float(np.ceil(lo)), float(np.floor(hi))
        bad = (s < lo) | (s > hi)
        if not bad.any():
            continue
        if opts.outliers == "cap":
            df[c] = df[c].clip(lo, hi)
            rep.log("Outliers", f"'{c}': {int(bad.sum())} extreme value(s) capped to [{lo:,.4g}, {hi:,.4g}]", int(bad.sum()))
        else:
            drop_mask |= bad
            rep.log("Outliers", f"'{c}': {int(bad.sum())} row(s) with extreme values flagged for removal", int(bad.sum()))
    if opts.outliers == "remove" and drop_mask.any():
        df = df[~drop_mask]
    return df


def _fill_missing(df: pd.DataFrame, opts: CleaningOptions, rep: CleaningReport) -> pd.DataFrame:
    df = df.copy()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if opts.numeric_missing == "drop":
        n = len(df)
        df = df.dropna(subset=num_cols)
        if n - len(df):
            rep.log("Missing numbers", "Dropped rows that still had missing numbers", n - len(df))
    elif opts.numeric_missing != "keep":
        for c in num_cols:
            n = int(df[c].isna().sum())
            if not n or ID_LIKE.search(c):
                continue
            value = {"median": df[c].median(), "mean": df[c].mean(), "zero": 0}[opts.numeric_missing]
            df[c] = df[c].fillna(value)
            rep.log("Missing numbers", f"'{c}': filled with the {opts.numeric_missing} ({value:,.2f})", n)
    if opts.text_missing:
        for c in df.columns:
            if _is_text(df[c]):
                n = int(df[c].isna().sum())
                if n:
                    df[c] = df[c].fillna(opts.text_missing)
                    rep.log("Missing text", f"'{c}': filled with '{opts.text_missing}'", n)
    return df


def _tidy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) and s.notna().all() and (s % 1 == 0).all():
            df[c] = s.astype("int64")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- public API
def clean_data(raw: pd.DataFrame, options: CleaningOptions | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    opts = options or CleaningOptions()
    rep = CleaningReport(rows_before=len(raw))

    df = _standardize_columns(raw, rep)
    df = _trim_and_nullify(df, rep)
    rep.missing_before = _missing_counts(df)          # includes "N/A"-style placeholders
    df = _drop_empty(df, rep)
    df = _infer_types(df, opts, rep)
    df = _standardize_text(df, opts, rep)
    if opts.drop_duplicates:
        df = _drop_duplicates(df, rep)
    if opts.negatives_invalid:
        df = _invalid_negatives(df, rep)
    df = _derive_missing_total(df, rep)
    df = _outliers(df, opts, rep)
    df = _fill_missing(df, opts, rep)
    df = _tidy_dtypes(df)

    rep.rows_after = len(df)
    rep.missing_after = _missing_counts(df)
    return df, rep
