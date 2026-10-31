"""
Stage 2 - ANALYSIS (SQL)

The cleaned DataFrame is loaded into an in-memory SQLite database (table: `data`)
and every number on the dashboard comes from a SQL query - aggregates, GROUP BY,
window functions (LAG, running totals, share of total) and CTEs.

Queries are generated from a "profile" of the dataset (which column is the
measure, which is the date, which are the categories), so they adapt to any CSV.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

MEASURE_HINTS = ("revenue", "sales", "amount", "total", "profit", "spend", "value", "cost", "price")
ID_LIKE = re.compile(r"(^|_)(id|uuid|key|index|sku|zip|pin|phone|mobile|code)$", re.I)
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I
)


# ----------------------------------------------------------------------------- profile
@dataclass
class Profile:
    measure: str | None = None
    date: str | None = None
    dims: list = field(default_factory=list)
    lo: float = 0.0
    hi: float = 0.0


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and not ID_LIKE.search(c)]


def date_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c]) and not ID_LIKE.search(c)]


def auto_profile(df: pd.DataFrame) -> Profile:
    nums, dates = numeric_columns(df), date_columns(df)
    measure = None
    for hint in MEASURE_HINTS:
        measure = next((c for c in nums if hint in c.lower()), None)
        if measure:
            break
    measure = measure or (nums[0] if nums else None)
    dims = [c for c in text_columns(df) if 2 <= df[c].nunique() <= 25][:3]
    return make_profile(df, measure, dates[0] if dates else None, dims)


def make_profile(df: pd.DataFrame, measure, date, dims) -> Profile:
    lo = hi = 0.0
    if measure is not None and df[measure].notna().any():
        lo, hi = float(df[measure].min()), float(df[measure].max())
    return Profile(measure, date, list(dims), lo, hi)


# ----------------------------------------------------------------------------- database
def load_to_sqlite(df: pd.DataFrame) -> sqlite3.Connection:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")        # ISO text -> works with strftime()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    out.to_sql("data", conn, index=False, if_exists="replace")
    conn.execute("PRAGMA query_only = ON")                   # read-only from here on
    return conn


def schema_of(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("PRAGMA table_info(data)").fetchall()
    return pd.DataFrame(rows, columns=["cid", "column", "sqlite_type", "notnull", "default", "pk"])[["column", "sqlite_type"]]


def run_sql(conn: sqlite3.Connection, sql: str, trusted: bool = False) -> pd.DataFrame:
    """Run one query. Untrusted (user-typed) SQL must be a single read-only SELECT."""
    sql = sql.strip().rstrip(";").strip()
    if not trusted:
        if ";" in sql:
            raise ValueError("Only one statement at a time, please.")
        if not re.match(r"(?is)^(select|with)\b", sql):
            raise ValueError("Only SELECT queries are allowed.")
        if FORBIDDEN.search(sql):
            raise ValueError("That statement type is not allowed (read-only playground).")
    return pd.read_sql_query(sql, conn)


# ----------------------------------------------------------------------------- query builders
def q(name: str) -> str:
    """Quote an identifier."""
    return '"' + name.replace('"', '""') + '"'


def lit(value) -> str:
    """Quote a string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def build_filter_conditions(date_col, date_range, dim_filters: dict) -> list:
    conds = []
    if date_col and date_range and len(date_range) == 2:
        conds.append(f"{q(date_col)} BETWEEN {lit(date_range[0])} AND {lit(date_range[1])}")
    for col, values in (dim_filters or {}).items():
        if values:
            conds.append(f"{q(col)} IN ({', '.join(lit(v) for v in values)})")
    return conds


def _where(conds, extra=()) -> str:
    parts = list(conds) + list(extra)
    return ("WHERE " + " AND ".join(parts)) if parts else ""


def build_queries(p: Profile, conds=None) -> dict:
    """Return {id: {title, description, sql}} for the dashboard + the SQL tab."""
    conds = conds or []
    Q: dict = {}

    def add(key, title, description, sql):
        Q[key] = {"title": title, "description": description, "sql": re.sub(r"\n\s*\n", "\n", sql.strip())}

    W = _where(conds)

    if p.measure is None:
        add("kpis", "Row count", "No numeric column found, so only counting rows.",
            f"SELECT COUNT(*) AS rows_count\nFROM data\n{W}")
        return Q

    M = q(p.measure)

    add("kpis", "Headline KPIs", "Totals and averages for the measure.",
        f"""SELECT COUNT(*)            AS rows_count,
       ROUND(SUM({M}), 2)   AS total,
       ROUND(AVG({M}), 2)   AS average,
       ROUND(MIN({M}), 2)   AS minimum,
       ROUND(MAX({M}), 2)   AS maximum
FROM data
{W}""")

    if p.date:
        D = q(p.date)
        Wd = _where(conds, [f"{D} IS NOT NULL"])
        add("monthly", "Monthly trend", "GROUP BY month using strftime().",
            f"""SELECT strftime('%Y-%m', {D}) AS month,
       ROUND(SUM({M}), 2)         AS total,
       COUNT(*)                   AS rows_count
FROM data
{Wd}
GROUP BY month
ORDER BY month""")

        add("mom", "Month-over-month growth + running total",
            "CTE + window functions LAG() and SUM() OVER (ORDER BY ...).",
            f"""WITH monthly AS (
    SELECT strftime('%Y-%m', {D}) AS month, SUM({M}) AS total
    FROM data
    {Wd}
    GROUP BY month
)
SELECT month,
       ROUND(total, 2) AS total,
       ROUND(100.0 * (total - LAG(total) OVER (ORDER BY month))
             / NULLIF(LAG(total) OVER (ORDER BY month), 0), 1) AS mom_growth_pct,
       ROUND(SUM(total) OVER (ORDER BY month), 2)              AS running_total
FROM monthly
ORDER BY month""")

    for d in p.dims[:3]:
        Dq = q(d)
        Wn = _where(conds, [f"{Dq} IS NOT NULL"])
        add(f"by_{d}", f"Performance by {d.replace('_', ' ')}",
            "GROUP BY with share-of-total via SUM(SUM(x)) OVER ().",
            f"""SELECT {Dq},
       ROUND(SUM({M}), 2) AS total,
       COUNT(*)           AS rows_count,
       ROUND(AVG({M}), 2) AS average,
       ROUND(100.0 * SUM({M}) / SUM(SUM({M})) OVER (), 1) AS share_pct
FROM data
{Wn}
GROUP BY {Dq}
ORDER BY total DESC
LIMIT 15""")

    if len(p.dims) >= 2:
        a, b = q(p.dims[0]), q(p.dims[1])
        Wc = _where(conds, [f"{a} IS NOT NULL", f"{b} IS NOT NULL"])
        add("cross", f"{p.dims[0].replace('_', ' ').title()} x {p.dims[1].replace('_', ' ')}",
            "Two-dimensional GROUP BY (feeds the heatmap).",
            f"""SELECT {a}, {b}, ROUND(SUM({M}), 2) AS total
FROM data
{Wc}
GROUP BY {a}, {b}
ORDER BY total DESC
LIMIT 200""")

    width = (p.hi - p.lo) / 10 if p.hi > p.lo else 1
    Wm = _where(conds, [f"{M} IS NOT NULL"])
    add("hist", f"Distribution of {p.measure.replace('_', ' ')}",
        "Equal-width buckets built with CAST() arithmetic.",
        f"""SELECT ROUND({p.lo} + MIN(CAST(({M} - {p.lo}) / {width} AS INTEGER), 9) * {width}, 2) AS bucket_start,
       COUNT(*) AS freq
FROM data
{Wm}
GROUP BY bucket_start
ORDER BY bucket_start""")

    add("top", f"Top 10 records by {p.measure.replace('_', ' ')}", "ORDER BY ... DESC LIMIT 10.",
        f"""SELECT *
FROM data
{Wm}
ORDER BY {M} DESC
LIMIT 10""")
    return Q
