# End-to-end data pipeline

**Messy CSV → pandas cleaning → SQL analysis → interactive dashboard**, in one Streamlit app.

```
 ┌────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌────────────────┐
 │ 1. IMPORT  │──▶│ 2. CLEAN (pandas)│──▶│ 3. ANALYSE (SQL)   │──▶│ 4. DASHBOARD   │
 │ upload CSV │   │ pipeline/        │   │ pipeline/analysis  │   │ app.py         │
 │ (all text) │   │ cleaning.py      │   │ SQLite, table data │   │ Plotly charts  │
 └────────────┘   └──────────────────┘   └────────────────────┘   └────────────────┘
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_sample_data.py                          # optional: creates sample_data/messy_sales.csv
streamlit run app.py
```

Open http://localhost:8501. With no upload, the app uses the built-in messy sales file.
Upload any CSV in the sidebar to run the pipeline on your own data.

## What each stage does

### 1 · Import
The CSV is read with every column as **text** (delimiter and encoding are auto-detected), so nothing is
silently mangled before cleaning.

### 2 · Cleaning (`pipeline/cleaning.py`, pandas)
| Problem | Fix |
|---|---|
| `" Order  Date "`, `Unit Price (INR)` | snake_case headers |
| Extra spaces | trimmed / collapsed |
| `N/A`, `null`, `-`, `?`, blanks | real missing values |
| Empty rows / columns | dropped |
| `₹1,200.50`, `Rs. 1200`, `(45)` | converted to numbers |
| `2025-01-31`, `31/01/2025`, `31-Jan-2025`, `January 31, 2025` | one datetime column |
| `north` / `NORTH` / ` North ` | one spelling |
| `Electronis` → `Electronics` | fuzzy merge (difflib) |
| Exact duplicates | removed (after standardising, so more are caught) |
| Negative quantity / price | treated as invalid → missing |
| `revenue` missing but quantity and price known | recomputed |
| Extreme outliers (`9999` units) | capped or removed (IQR; skewed columns are judged on a log scale) |
| Remaining gaps | median / mean / 0 / drop / leave; text → `Unknown` |

Every rule is toggleable in the sidebar, and a **cleaning report** lists exactly what changed and how many cells it touched.

### 3 · Analysis (`pipeline/analysis.py`, SQLite)
The cleaned frame is loaded into an in-memory SQLite table named `data`. All dashboard numbers come from SQL:
aggregates, `GROUP BY`, `strftime`, CTEs, window functions (`LAG` for month-over-month growth, running totals,
`SUM(SUM(x)) OVER ()` for share of total). Queries are generated from the detected measure / date / category
columns (override them in the sidebar), and they respect the dashboard filters. The **SQL analysis** tab shows
each query next to its result, and the **SQL playground** lets you run your own read-only `SELECT`s.

### 4 · Dashboard (`app.py`, Plotly)
KPI cards, monthly trend, month-over-month growth, running total, breakdown bars + donut charts, a
category × category heatmap, a distribution histogram and a top-10 table, all filterable by date and category.

## Project layout
```
app.py                    Streamlit UI (the 5 tabs)
pipeline/cleaning.py      Stage 2: pandas cleaning + report
pipeline/analysis.py      Stage 3: SQLite loading + SQL query builder
generate_sample_data.py   Creates a deliberately messy demo CSV
sample_data/              Demo CSV
```

## Good to know
* **Day-first dates** (31/12/2025) are assumed by default. Untick the box for US-style month-first data.
* Numbers like `1.234,56` (European format) are not supported, since commas are treated as thousands separators.
* The pipeline holds data in memory, which is comfortable up to a few hundred thousand rows.
  For bigger files swap SQLite for DuckDB (`duckdb.connect()` + `conn.register("data", df)`).
* Outlier capping and "negative = invalid" are sensible defaults, not universal truths. Review the cleaning report
  and switch them off if your data legitimately has refunds or extreme values.

## Extending it
* **New cleaning rule**: add a `_function(df, rep)` in `cleaning.py`, call it inside `clean_data`, and `rep.log(...)` what it did.
* **New chart**: add a query in `build_queries()`, then plot `result("your_key")` in the Dashboard tab.
