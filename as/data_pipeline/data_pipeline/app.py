
"""
END-TO-END DATA PIPELINE APP

CSV Upload -> Pandas Cleaning -> SQL Analysis -> Interactive Dashboard

Run: streamlit run app.py
"""

from __future__ import annotations

# ==========================================================
# 1. AUTOMATICALLY INSTALL REQUIRED LIBRARIES
# ==========================================================

import sys
import subprocess
import importlib.util

required_packages = [
    "pandas",
    "plotly",
    "streamlit",
    "packaging"
]

for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package
        ])

print("All required libraries are ready!")

# ==========================================================
# 2. IMPORT LIBRARIES
# ==========================================================

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from packaging.version import Version

from pipeline.analysis import (
    auto_profile,
    build_filter_conditions,
    build_queries,
    date_columns,
    load_to_sqlite,
    make_profile,
    numeric_columns,
    run_sql,
    schema_of,
    text_columns,
)

from pipeline.cleaning import CleaningOptions, clean_data


# ==========================================================
# 3. SETTINGS
# ==========================================================

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "messy_sales.csv"

PALETTE = px.colors.qualitative.Set2

# Streamlit version compatibility
STRETCH = (
    {"width": "stretch"}
    if Version(st.__version__) >= Version("1.50.0")
    else {"use_container_width": True}
)

st.set_page_config(
    page_title="Data Pipeline",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# 4. CACHED FUNCTIONS
# ==========================================================

@st.cache_data(show_spinner=False)
def read_csv(data: bytes) -> pd.DataFrame:

    """Read everything as text."""

    return pd.read_csv(
        io.BytesIO(data),
        dtype=str,
        keep_default_na=False,
        sep=None,
        engine="python",
        encoding_errors="replace",
        on_bad_lines="skip",
    )


@st.cache_data(show_spinner=False)
def cached_clean(raw: pd.DataFrame, opts: CleaningOptions):

    return clean_data(raw, opts)


@st.cache_resource(show_spinner=False)
def get_connection(_df: pd.DataFrame, fingerprint: int):

    return load_to_sqlite(_df)


# ==========================================================
# 5. HELPER FUNCTIONS
# ==========================================================

def nice(col: str) -> str:

    return col.replace("_", " ").title()


def fmt(v) -> str:

    if v is None or pd.isna(v):
        return "—"

    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def show(fig):

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text=""
    )

    st.plotly_chart(fig, **STRETCH)


# ==========================================================
# 6. MAIN TITLE
# ==========================================================

st.title("📊 End-to-end Data Pipeline")

st.caption(
    "Messy CSV → Pandas Cleaning → SQL Analysis → Dashboard"
)


# ==========================================================
# 7. SIDEBAR: DATA IMPORT AND CLEANING RULES
# ==========================================================

with st.sidebar:

    st.header("1 · Import")

    uploaded = st.file_uploader(
        "Upload a messy CSV",
        type=["csv", "txt"]
    )

    st.header("2 · Cleaning Rules")

    dayfirst = st.checkbox(
        "Dates are day-first (31/12/2025)",
        value=True,
        help="Untick for US-style month-first dates."
    )

    fuzzy = st.checkbox(
        "Merge typos in labels",
        value=True,
        help="Example: Electronis → Electronics"
    )

    dedup = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    negatives = st.checkbox(
        "Negative qty/price/amount = invalid",
        value=True
    )

    outliers = st.selectbox(
        "Extreme outliers",
        ["cap", "remove", "keep"],
        format_func={
            "cap": "Cap to a sensible limit",
            "remove": "Remove rows",
            "keep": "Keep as is"
        }.get
    )

    num_missing = st.selectbox(
        "Missing numbers",
        ["median", "mean", "zero", "drop", "keep"],
        format_func={
            "median": "Fill with median",
            "mean": "Fill with mean",
            "zero": "Fill with 0",
            "drop": "Drop the rows",
            "keep": "Leave empty"
        }.get
    )


# ==========================================================
# 8. LOAD DATA
# ==========================================================

if uploaded is not None:

    raw = read_csv(uploaded.getvalue())

    source_name = uploaded.name

elif SAMPLE_PATH.exists():

    raw = read_csv(SAMPLE_PATH.read_bytes())

    source_name = "sample_data/messy_sales.csv (built-in demo)"

    st.info(
        "No file uploaded yet - showing the built-in messy sales dataset. "
        "Upload your own CSV in the sidebar."
    )

else:

    st.warning(
        "Upload a CSV in the sidebar to begin."
    )

    st.stop()


# ==========================================================
# 9. DATA CLEANING
# ==========================================================

options = CleaningOptions(

    dayfirst=dayfirst,

    fuzzy_merge=fuzzy,

    drop_duplicates=dedup,

    negatives_invalid=negatives,

    outliers=outliers,

    numeric_missing=num_missing,
)


with st.spinner("Cleaning data with pandas..."):

    clean, report = cached_clean(raw, options)


if clean.empty:

    st.error(
        "Nothing left after cleaning. "
        "Try relaxing the cleaning rules in the sidebar."
    )

    st.stop()


# ==========================================================
# 10. ANALYSIS SETTINGS AND FILTERS
# ==========================================================

default = auto_profile(clean)

with st.sidebar:

    st.header("3 · Analysis Settings")

    nums = numeric_columns(clean)

    measure = (
        st.selectbox(
            "Measure (what to add up)",
            nums,
            index=nums.index(default.measure)
        )
        if nums else None
    )

    dates = date_columns(clean)

    date_choice = st.selectbox(
        "Date column",
        ["(none)"] + dates,
        index=(
            dates.index(default.date) + 1
            if default.date in dates
            else 0
        )
    )

    date_col = (
        None if date_choice == "(none)"
        else date_choice
    )

    texts = text_columns(clean)

    dims = st.multiselect(
        "Categories to break down by",
        texts,
        default=[
            d for d in default.dims if d in texts
        ][:3],
        max_selections=3
    )

    profile = make_profile(
        clean,
        measure,
        date_col,
        dims
    )

    st.header("4 · Dashboard Filters")

    date_range = None

    if date_col and clean[date_col].notna().any():

        dmin = clean[date_col].min().date()

        dmax = clean[date_col].max().date()

        picked = st.date_input(
            "Date range",
            (dmin, dmax),
            min_value=dmin,
            max_value=dmax
        )

        if isinstance(picked, (list, tuple)) and len(picked) == 2:

            date_range = (
                str(picked[0]),
                str(picked[1])
            )

    dim_filters = {}

    for d in profile.dims:

        chosen = st.multiselect(
            nice(d),
            sorted(clean[d].dropna().unique())
        )

        if chosen:

            dim_filters[d] = chosen


# ==========================================================
# 11. SQL CONNECTION AND QUERIES
# ==========================================================

fingerprint = int(
    pd.util.hash_pandas_object(
        clean,
        index=True
    ).sum() % (2 ** 62)
)

conn = get_connection(clean, fingerprint)

conds = build_filter_conditions(
    profile.date,
    date_range,
    dim_filters
)

queries = build_queries(
    profile,
    conds
)


def result(key: str) -> pd.DataFrame:

    return run_sql(
        conn,
        queries[key]["sql"],
        trusted=True
    )


# ==========================================================
# 12. DASHBOARD TABS
# ==========================================================

tab_raw, tab_clean, tab_sql, tab_dash, tab_play = st.tabs(
    [
        "① Raw Data",
        "② Cleaning",
        "③ SQL Analysis",
        "④ Dashboard",
        "⑤ SQL Playground"
    ]
)


# ==========================================================
# TAB 1: RAW DATA
# ==========================================================

with tab_raw:

    st.subheader(f"Raw Import: {source_name}")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{len(raw):,}"
    )

    c2.metric(
        "Columns",
        raw.shape[1]
    )

    c3.metric(
        "Exact Duplicate Rows",
        f"{int(raw.duplicated().sum()):,}"
    )

    st.dataframe(
        raw.head(200),
        **STRETCH,
        height=420
    )

    st.caption(
        "Everything is read as text on purpose. "
        "The cleaning stage decides the real data types."
    )


# ==========================================================
# TAB 2: CLEANING
# ==========================================================

with tab_clean:

    st.subheader("What the Cleaning Stage Did")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows In",
        f"{report.rows_before:,}"
    )

    c2.metric(
        "Rows Out",
        f"{report.rows_after:,}",
        f"{report.rows_after - report.rows_before:,}",
        delta_color="off"
    )

    c3.metric(
        "Fixes Applied",
        f"{report.total_fixes:,}"
    )

    c4.metric(
        "Missing Cells Left",
        f"{sum(report.missing_after.values()):,}"
    )

    left, right = st.columns([3, 2])

    with left:

        st.dataframe(
            report.steps_frame(),
            **STRETCH,
            hide_index=True,
            height=420
        )

    with right:

        miss = pd.DataFrame({

            "column": list(report.missing_before),

            "Before": list(report.missing_before.values()),

        })

        miss["After"] = (
            miss["column"]
            .map(report.missing_after)
            .fillna(0)
            .astype(int)
        )

        miss = miss[
            (miss["Before"] > 0) |
            (miss["After"] > 0)
        ].melt(
            "column",
            var_name="stage",
            value_name="missing cells"
        )

        if not miss.empty:

            show(
                px.bar(
                    miss,
                    x="column",
                    y="missing cells",
                    color="stage",
                    barmode="group",
                    title="Missing Values Before vs After",
                    color_discrete_sequence=PALETTE
                )
            )

        else:

            st.success(
                "No missing values were found."
            )

    st.subheader("Cleaned Data")

    st.dataframe(
        clean.head(200),
        **STRETCH,
        height=320
    )

    st.download_button(
        "⬇️ Download Cleaned CSV",
        clean.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )


# ==========================================================
# TAB 3: SQL ANALYSIS
# ==========================================================

with tab_sql:

    st.subheader("Analysis Queries (SQLite)")

    st.caption(
        "The cleaned data lives in a table called data. "
        "Every dashboard number is the result of one of these queries "
        "and responds to the sidebar filters."
    )

    for key, qd in queries.items():

        with st.expander(
            f"{qd['title']} - {qd['description']}",
            expanded=(key == "kpis")
        ):

            st.code(
                qd["sql"],
                language="sql"
            )

            st.dataframe(
                result(key),
                **STRETCH,
                hide_index=True
            )


# ==========================================================
# TAB 4: DASHBOARD
# ==========================================================

with tab_dash:

    k = result("kpis").iloc[0]

    if int(k["rows_count"]) == 0:

        st.warning(
            "No rows match the current filters."
        )

    elif profile.measure is None:

        st.metric(
            "Rows",
            f"{int(k['rows_count']):,}"
        )

        st.warning(
            "This file has no numeric column, "
            "so there is nothing to total or chart. "
            "Check the cleaning report to see whether "
            "numbers were detected."
        )

    else:

        m = nice(profile.measure)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            f"Total {m}",
            fmt(k["total"])
        )

        c2.metric(
            f"Average {m}",
            fmt(k["average"])
        )

        c3.metric(
            "Records",
            f"{int(k['rows_count']):,}"
        )

        c4.metric(
            f"Largest {m}",
            fmt(k["maximum"])
        )

        st.divider()

        # ------------------------------------------
        # TIME ANALYSIS
        # ------------------------------------------

        if "monthly" in queries:

            mom = result("mom")

            if not mom.empty:

                a, b = st.columns([3, 2])

                with a:

                    fig = px.area(
                        mom,
                        x="month",
                        y="total",
                        title=f"{m} by Month",
                        color_discrete_sequence=[PALETTE[0]]
                    )

                    show(fig)

                with b:

                    g = mom.dropna(
                        subset=["mom_growth_pct"]
                    )

                    if not g.empty:

                        g = g.assign(
                            direction=g["mom_growth_pct"]
                            .ge(0)
                            .map({
                                True: "Growth",
                                False: "Decline"
                            })
                        )

                        show(
                            px.bar(
                                g,
                                x="month",
                                y="mom_growth_pct",
                                color="direction",
                                title="Month-over-Month Growth (%)",
                                color_discrete_map={
                                    "Growth": "#2e9d6a",
                                    "Decline": "#d9534f"
                                }
                            )
                        )

                show(
                    px.line(
                        mom,
                        x="month",
                        y="running_total",
                        title=f"Running Total of {m}",
                        color_discrete_sequence=[PALETTE[2]]
                    )
                )

        # ------------------------------------------
        # CATEGORY ANALYSIS
        # ------------------------------------------

        for i, d in enumerate(profile.dims[:2]):

            df_d = result(f"by_{d}")

            if df_d.empty:

                continue

            a, b = st.columns([3, 2])

            with a:

                fig = px.bar(
                    df_d,
                    x="total",
                    y=d,
                    orientation="h",
                    text="share_pct",
                    title=f"{m} by {nice(d)}",
                    color_discrete_sequence=[
                        PALETTE[i % len(PALETTE)]
                    ]
                )

                fig.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside"
                )

                fig.update_layout(
                    yaxis=dict(
                        autorange="reversed",
                        title=""
                    ),
                    xaxis_title=""
                )

                show(fig)

            with b:

                show(
                    px.pie(
                        df_d.head(8),
                        names=d,
                        values="total",
                        hole=0.5,
                        title=f"Share of {m}: {nice(d)}",
                        color_discrete_sequence=PALETTE
                    )
                )

        # ------------------------------------------
        # CROSS ANALYSIS AND DISTRIBUTION
        # ------------------------------------------

        a, b = st.columns(2)

        with a:

            if "cross" in queries:

                cr = result("cross")

                if not cr.empty:

                    d1, d2 = profile.dims[:2]

                    pv = cr.pivot_table(
                        index=d1,
                        columns=d2,
                        values="total",
                        aggfunc="sum"
                    )

                    show(
                        px.imshow(
                            pv,
                            text_auto=".2s",
                            aspect="auto",
                            color_continuous_scale="Blues",
                            title=f"{nice(d1)} × {nice(d2)} ({m})"
                        )
                    )

        with b:

            hist = result("hist")

            if not hist.empty:

                show(
                    px.bar(
                        hist,
                        x="bucket_start",
                        y="freq",
                        title=f"Distribution of {m}",
                        labels={
                            "bucket_start": m,
                            "freq": "records"
                        },
                        color_discrete_sequence=[PALETTE[3]]
                    )
                )

        # ------------------------------------------
        # TOP 10 RECORDS
        # ------------------------------------------

        st.subheader(f"Top 10 Records by {m}")

        st.dataframe(
            result("top"),
            **STRETCH,
            hide_index=True
        )


# ==========================================================
# TAB 5: SQL PLAYGROUND
# ==========================================================

with tab_play:

    st.subheader("Ask Your Own Questions in SQL")

    a, b = st.columns([1, 3])

    with a:

        st.caption("Table: data")

        st.dataframe(
            schema_of(conn),
            hide_index=True,
            **STRETCH
        )

    with b:

        example = (
            f'SELECT "{profile.dims[0]}", COUNT(*) AS n\n'
            f'FROM data\n'
            f'GROUP BY "{profile.dims[0]}"\n'
            f'ORDER BY n DESC'
            if profile.dims
            else "SELECT * FROM data LIMIT 20"
        )

        sql = st.text_area(
            "SQL (read-only, one SELECT statement)",
            value=example,
            height=160
        )

        if st.button(
            "▶ Run Query",
            type="primary"
        ):

            try:

                out = run_sql(
                    conn,
                    sql
                )

                st.success(
                    f"{len(out):,} row(s)"
                )

                st.dataframe(
                    out,
                    **STRETCH,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Result",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="query_result.csv",
                    mime="text/csv"
                )

            except Exception as exc:

                st.error(
                    f"SQL error: {exc}"
                )
