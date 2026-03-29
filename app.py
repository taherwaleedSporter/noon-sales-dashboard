```python
import io
from pathlib import Path
import pandas as pd
import streamlit as st

VALID_STATUSES = {"Processing", "Shipped", "Delivered"}

st.set_page_config(page_title="Noon Sales Dashboard", layout="wide")

def open_local(path: str):
    return open(path, "rb") if Path(path).exists() else None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def read_orders(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()

    df = pd.read_csv(file) if str(file.name).endswith(".csv") else pd.read_excel(file)
    df = standardize_columns(df)

    if "currency_code" in df.columns and "currency" not in df.columns:
        df["currency"] = df["currency_code"]

    df = df[df["status"].isin(VALID_STATUSES)].copy()

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["offer_price"] = pd.to_numeric(df["offer_price"], errors="coerce")

    df["order_datetime"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
    df = df.dropna(subset=["order_datetime", "partner_sku", "offer_price"])

    df["date_dt"] = pd.to_datetime(df["order_datetime"].dt.date)
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    df["month"] = df["date_dt"].dt.strftime("%Y-%m")

    return df

def read_reference(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()

    df = pd.read_csv(file) if str(file.name).endswith(".csv") else pd.read_excel(file)
    df = standardize_columns(df)

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["cogs_aed"] = pd.to_numeric(df["cogs_aed"], errors="coerce")

    df["fee_percent"] = (
        df["fee_percent"].astype(str).str.replace("%", "").astype(float) / 100
    )

    return df

def read_fx(file) -> pd.DataFrame:
    df = pd.read_csv(file) if str(file.name).endswith(".csv") else pd.read_excel(file)
    df = standardize_columns(df)
    df["to_usd"] = pd.to_numeric(df["to_usd"], errors="coerce")
    return df

def read_stock(file) -> pd.DataFrame:
    df = pd.read_csv(file) if str(file.name).endswith(".csv") else pd.read_excel(file)
    df = standardize_columns(df)

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)

    return df.groupby("partner_sku", as_index=False).agg(
        current_stock_qty=("qty", "sum")
    )

@st.cache_data
def build_model(orders_df, ref_df, fx_df):
    model = orders_df.merge(ref_df, on="partner_sku", how="left")
    model = model.merge(fx_df, on="currency", how="left")

    model["sales_usd"] = model["offer_price"] * model["to_usd"]
    model["cogs_usd"] = model["cogs_aed"] * fx_df.loc[fx_df["currency"] == "AED", "to_usd"].iloc[0]
    model["gp_usd"] = model["sales_usd"] - model["cogs_usd"]

    return model

def classify_stock_risk(days_of_cover, stock):
    if stock <= 0:
        return "Out of stock"
    if days_of_cover < 14:
        return "Critical"
    if days_of_cover <= 30:
        return "Warning"
    return "Healthy"

def make_stock_summary(df, stock_df):
    sales = df.groupby("partner_sku")["partner_sku"].count().reset_index(name="orders")
    out = stock_df.merge(sales, on="partner_sku", how="left").fillna(0)

    out["avg_daily"] = out["orders"] / 14
    out["days_of_cover"] = out["current_stock_qty"] / out["avg_daily"]

    out["stock_risk"] = out.apply(
        lambda r: classify_stock_risk(r["days_of_cover"], r["current_stock_qty"]),
        axis=1,
    )

    return out

# ✅ FIXED STYLE FUNCTION
def style_stock_coverage(df: pd.DataFrame):
    def row_style(row):
        risk = row.get("stock_risk")

        if risk == "Out of stock":
            style = "background-color: #f8d7da; color: black;"
        elif risk == "Critical":
            style = "background-color: #f8d7da; color: black;"
        elif risk == "Warning":
            style = "background-color: #fff3cd; color: black;"
        elif risk == "Healthy":
            style = "background-color: #d1e7dd; color: black;"
        else:
            style = ""

        return [style] * len(row)

    return df.style.apply(row_style, axis=1)

# UI
st.title("Noon Dashboard")

orders_file = st.file_uploader("Orders")
ref_file = st.file_uploader("Reference")
fx_file = st.file_uploader("FX")
stock_file = st.file_uploader("Stock")

if orders_file and ref_file and fx_file and stock_file:
    orders = read_orders(orders_file)
    ref = read_reference(ref_file)
    fx = read_fx(fx_file)
    stock = read_stock(stock_file)

    model = build_model(orders, ref, fx)
    stock_cov = make_stock_summary(model, stock)

    st.subheader("Stock Coverage")
    st.dataframe(style_stock_coverage(stock_cov), use_container_width=True)
```
