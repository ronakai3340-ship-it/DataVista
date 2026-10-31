"""Generate a deliberately MESSY sales CSV to test the pipeline.

Run:  python generate_sample_data.py
Out:  sample_data/messy_sales.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 900

CATALOG = {
    "Electronics": {"Laptop": 62000, "Smartphone": 28000, "Headphones": 3500, "Smartwatch": 9000},
    "Furniture":   {"Office Chair": 7500, "Desk": 12000, "Bookshelf": 5500},
    "Stationery":  {"Notebook": 120, "Pen Set": 350, "Planner": 600},
    "Clothing":    {"T-Shirt": 700, "Jeans": 1800, "Jacket": 4200},
}
REGIONS = ["North", "South", "East", "West"]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
FIRST = ["Aarav", "Diya", "Rohan", "Isha", "Kabir", "Meera", "Vivaan", "Anaya", "Arjun", "Saanvi",
         "Reyansh", "Kiara", "Dev", "Myra", "Yash", "Tara"]
LAST = ["Sharma", "Patel", "Iyer", "Khan", "Gupta", "Nair", "Mehta", "Reddy", "Singh", "Joshi"]

rows = []
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 330, N), unit="D")
for i in range(N):
    cat = rng.choice(list(CATALOG), p=[0.35, 0.2, 0.25, 0.2])
    product = rng.choice(list(CATALOG[cat]))
    price = CATALOG[cat][product] * rng.uniform(0.9, 1.1)
    # growth + festive spike so the trend chart has a story
    qty = int(rng.integers(1, 6) + (2 if dates[i].month in (10, 11) else 0))
    rows.append({
        "Order ID": f"ORD-{1000 + i}",
        "Customer Name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "Order Date": dates[i],
        "Product Category": cat,
        "Product": product,
        "Region": rng.choice(REGIONS, p=[0.3, 0.25, 0.2, 0.25]),
        "Quantity": qty,
        "Unit Price (INR)": round(price, 2),
        "Payment Method": rng.choice(PAYMENTS, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
    })
df = pd.DataFrame(rows)
df["Revenue"] = (df["Quantity"] * df["Unit Price (INR)"]).round(2)

# ------------------------------------------------------------------ now make it messy
out = df.astype(object).copy()

# 1) messy header whitespace
out = out.rename(columns={"Customer Name": " Customer Name ", "Order Date": "Order  Date"})

# 2) mixed date formats
fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%B %d, %Y", "%d %b %Y"]
out["Order  Date"] = [d.strftime(fmts[rng.integers(0, len(fmts))]) for d in df["Order Date"]]

# 3) currency formatting
def money(v):
    style = rng.integers(0, 4)
    return [f"{v:.2f}", f"₹{v:,.2f}", f"Rs. {v:,.0f}", f"{v:,.2f}"][style]
out["Unit Price (INR)"] = [money(v) for v in df["Unit Price (INR)"]]
out["Revenue"] = [money(v) for v in df["Revenue"]]

# 4) inconsistent casing / spaces / typos in categories
def jitter(v):
    r = rng.random()
    if r < 0.12: return v.upper()
    if r < 0.24: return v.lower()
    if r < 0.32: return f"  {v} "
    return v
for col in ["Region", "Payment Method", "Product Category"]:
    out[col] = [jitter(v) for v in out[col]]
typo_idx = rng.choice(N, 14, replace=False)
out.loc[typo_idx, "Product Category"] = "Electronis"
out.loc[rng.choice(N, 8, replace=False), "Product Category"] = "Furnture"

# 5) missing values in different disguises
for col, frac in [("Region", 0.05), ("Quantity", 0.04), ("Revenue", 0.07), ("Payment Method", 0.04),
                  ("Unit Price (INR)", 0.03), ("Order  Date", 0.02), ("Product", 0.02)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    out.loc[idx, col] = rng.choice(["", "N/A", "null", "-", "?", "NA"], len(idx))

# 6) impossible values and outliers
out.loc[rng.choice(N, 6, replace=False), "Quantity"] = -3
out.loc[rng.choice(N, 3, replace=False), "Quantity"] = 9999
out.loc[rng.choice(N, 3, replace=False), "Unit Price (INR)"] = "99999999"

# 7) duplicate rows + a fully empty row
out = pd.concat([out, out.sample(35, random_state=1)], ignore_index=True)
out.loc[len(out)] = [""] * out.shape[1]
out = out.sample(frac=1, random_state=7).reset_index(drop=True)

path = Path(__file__).parent / "sample_data" / "messy_sales.csv"
path.parent.mkdir(exist_ok=True)
out.to_csv(path, index=False)
print(f"Wrote {len(out)} messy rows -> {path}")
