
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

    name = getattr(file, "name", "")
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = standardize_columns(df)

    # Handle alternate column names
    if "currency_code" in df.columns and "currency" not in df.columns:
        df["currency"] = df["currency_code"]

    required = {"partner_sku", "status", "order_timestamp", "currency", "offer_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Orders file is missing columns: {', '.join(sorted(missing))}")

    df["status"] = df["status"].astype(str).str.strip()
    df = df[df["status"].isin(VALID_STATUSES)].copy()

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["offer_price"] = pd.to_numeric(df["offer_price"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.strip()

    df["order_datetime"] = pd.to_datetime(
        df["order_timestamp"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    df = df.dropna(subset=["order_datetime", "partner_sku", "offer_price"]).copy()

    df["date_dt"] = pd.to_datetime(df["order_datetime"].dt.date)
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    df["month"] = df["date_dt"].dt.strftime("%Y-%m")

    return df


def read_reference(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["partner_sku", "product_title", "brand", "cogs_aed", "fee_percent"])

    name = getattr(file, "name", "")
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file, encoding="latin1")
    else:
        df = pd.read_excel(file)

    df = standardize_columns(df)

    required = {"partner_sku", "product_title", "brand", "cogs_aed", "fee_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Product reference file is missing columns: {', '.join(sorted(missing))}")

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["cogs_aed"] = pd.to_numeric(df["cogs_aed"], errors="coerce")

    df["fee_percent"] = df["fee_percent"].astype(str).str.replace("%", "", regex=False).str.strip()
    df["fee_percent"] = pd.to_numeric(df["fee_percent"], errors="coerce")
    df["fee_percent"] = df["fee_percent"].apply(lambda x: x / 100.0 if pd.notna(x) and x > 1 else x)

    return df


def read_fx(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["currency", "to_usd"])

    name = getattr(file, "name", "")
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = standardize_columns(df)

    required = {"currency", "to_usd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FX file is missing columns: {', '.join(sorted(missing))}")

    df["currency"] = df["currency"].astype(str).str.strip()
    df["to_usd"] = pd.to_numeric(df["to_usd"], errors="coerce")

    return df.dropna(subset=["currency", "to_usd"]).copy()


def read_stock(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["partner_sku", "current_stock_qty", "inventory_snapshot_at"])

    name = getattr(file, "name", "")
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = standardize_columns(df)

    required = {"partner_sku", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stock file is missing columns: {', '.join(sorted(missing))}")

    df["partner_sku"] = pd.to_numeric(df["partner_sku"], errors="coerce").astype("Int64")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)

    if "inventory_snapshot_at" in df.columns:
        df["inventory_snapshot_at"] = pd.to_datetime(df["inventory_snapshot_at"], errors="coerce")
    else:
        df["inventory_snapshot_at"] = pd.NaT

    text_fallbacks = {}
    for col in ["title", "brand", "sku"]:
        if col in df.columns:
            text_fallbacks[col] = lambda s=df[col]: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else None

    grouped = (
        df.groupby("partner_sku", dropna=False, as_index=False)
        .agg(
            current_stock_qty=("qty", "sum"),
            inventory_snapshot_at=("inventory_snapshot_at", "max"),
            stock_title=("title", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else None) if "title" in df.columns else ("qty", "size"),
            stock_brand=("brand", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else None) if "brand" in df.columns else ("qty", "size"),
            stock_sku=("sku", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else None) if "sku" in df.columns else ("qty", "size"),
        )
    )

    for col in ["stock_title", "stock_brand", "stock_sku"]:
        if col in grouped.columns:
            grouped[col] = grouped[col].replace({0: None})

    grouped["inventory_snapshot_at"] = pd.to_datetime(grouped["inventory_snapshot_at"], errors="coerce")
    return grouped.dropna(subset=["partner_sku"]).copy()


@st.cache_data
def build_model(orders_df: pd.DataFrame, ref_df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    model = orders_df.merge(ref_df, on="partner_sku", how="left")
    model = model.merge(fx_df, on="currency", how="left")

    aed_rate_series = fx_df.loc[fx_df["currency"].eq("AED"), "to_usd"]
    if aed_rate_series.empty:
        raise ValueError("FX table must contain an AED row so COGS can be converted to USD.")

    aed_to_usd = float(aed_rate_series.iloc[0])

    model["sales_usd"] = model["offer_price"] * model["to_usd"]
    model["cogs_usd"] = model["cogs_aed"] * aed_to_usd
    model["fees_usd"] = model["sales_usd"] * model["fee_percent"]
    model["gp_usd"] = model["sales_usd"] - model["cogs_usd"] - model["fees_usd"]
    model["net_gp_usd"] = model["gp_usd"]
    model["orders"] = 1
    model["margin_pct"] = model["gp_usd"] / model["sales_usd"]

    return model


def make_daily_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["date", "brand"], dropna=False, as_index=False)
        .agg(
            sales_usd=("sales_usd", "sum"),
            orders=("orders", "sum"),
            cogs=("cogs_usd", "sum"),
            estimated_fees=("fees_usd", "sum"),
            gp=("gp_usd", "sum"),
            net_gp=("net_gp_usd", "sum"),
        )
        .sort_values(["date", "brand"])
    )


def make_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "brand"], dropna=False, as_index=False)
        .agg(
            sales_usd=("sales_usd", "sum"),
            orders=("orders", "sum"),
            cogs=("cogs_usd", "sum"),
            estimated_fees=("fees_usd", "sum"),
            gp=("gp_usd", "sum"),
            net_gp=("net_gp_usd", "sum"),
        )
        .sort_values(["month", "brand"])
    )


def make_product_orders(df: pd.DataFrame) -> pd.DataFrame:
    monthly_cols = sorted(df["month"].dropna().unique().tolist())

    out = pd.pivot_table(
        df,
        index=["partner_sku", "product_title", "brand"],
        columns="month",
        values="orders",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    out.columns.name = None

    for c in monthly_cols:
        if c not in out.columns:
            out[c] = 0

    out["total"] = out[monthly_cols].sum(axis=1) if monthly_cols else 0
    out = out[["partner_sku", "product_title", "brand", "total", *monthly_cols]]

    return out.sort_values(["brand", "partner_sku"])


def make_product_sales(df: pd.DataFrame) -> pd.DataFrame:
    monthly_cols = sorted(df["month"].dropna().unique().tolist())

    out = pd.pivot_table(
        df,
        index=["partner_sku", "product_title", "brand"],
        columns="month",
        values="sales_usd",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    out.columns.name = None

    for c in monthly_cols:
        if c not in out.columns:
            out[c] = 0.0

    out["total"] = out[monthly_cols].sum(axis=1) if monthly_cols else 0.0
    out = out[["partner_sku", "product_title", "brand", "total", *monthly_cols]]

    return out.sort_values(["brand", "partner_sku"])


def make_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["partner_sku", "product_title", "brand"], dropna=False, as_index=False)
        .agg(
            total_sales_usd=("sales_usd", "sum"),
            total_orders=("orders", "sum"),
            cogs_usd=("cogs_usd", "sum"),
            estimated_fees=("fees_usd", "sum"),
            gp_usd=("gp_usd", "sum"),
        )
        .sort_values("total_sales_usd", ascending=False)
    )
    out["margin_pct"] = out["gp_usd"] / out["total_sales_usd"]
    return out


def make_gainers_decliners(df: pd.DataFrame):
    latest_date = df["date_dt"].max()
    if pd.isna(latest_date):
        return pd.DataFrame(), pd.DataFrame()

    latest_date = pd.Timestamp(latest_date)
    last_7_start = latest_date - pd.Timedelta(days=6)
    prev_7_start = latest_date - pd.Timedelta(days=13)
    prev_7_end = latest_date - pd.Timedelta(days=7)

    last_7 = df[(df["date_dt"] >= last_7_start) & (df["date_dt"] <= latest_date)].copy()
    prev_7 = df[(df["date_dt"] >= prev_7_start) & (df["date_dt"] <= prev_7_end)].copy()

    last_7_sales = (
        last_7.groupby(["partner_sku", "product_title", "brand"], dropna=False)["sales_usd"]
        .sum()
        .reset_index(name="sales_last_7d")
    )

    prev_7_sales = (
        prev_7.groupby(["partner_sku", "product_title", "brand"], dropna=False)["sales_usd"]
        .sum()
        .reset_index(name="sales_prev_7d")
    )

    merged = last_7_sales.merge(
        prev_7_sales,
        on=["partner_sku", "product_title", "brand"],
        how="outer",
    ).fillna(0)

    merged["sales_change"] = merged["sales_last_7d"] - merged["sales_prev_7d"]

    top_gainers = merged.sort_values("sales_change", ascending=False).head(10)
    top_decliners = merged.sort_values("sales_change", ascending=True).head(10)

    return top_gainers, top_decliners


def make_mtd_comparison(df: pd.DataFrame):
    latest_date = df["date_dt"].max()
    if pd.isna(latest_date):
        return None

    latest_date = pd.Timestamp(latest_date)
    current_month_start = latest_date.replace(day=1)
    previous_month_start = (current_month_start - pd.DateOffset(months=1)).replace(day=1)

    previous_month_end = previous_month_start + pd.offsets.MonthEnd(0)
    previous_same_day = previous_month_start + pd.DateOffset(days=latest_date.day - 1)
    previous_same_day = min(previous_same_day, previous_month_end)

    mtd = df[(df["date_dt"] >= current_month_start) & (df["date_dt"] <= latest_date)].copy()
    prev_mtd = df[(df["date_dt"] >= previous_month_start) & (df["date_dt"] <= previous_same_day)].copy()

    mtd_sales = float(mtd["sales_usd"].sum())
    prev_sales = float(prev_mtd["sales_usd"].sum())
    mtd_orders = int(mtd["orders"].sum())
    prev_orders = int(prev_mtd["orders"].sum())
    mtd_gp = float(mtd["gp_usd"].sum())
    prev_gp = float(prev_mtd["gp_usd"].sum())

    sales_delta = ((mtd_sales - prev_sales) / prev_sales) if prev_sales else None
    orders_delta = ((mtd_orders - prev_orders) / prev_orders) if prev_orders else None
    gp_delta = ((mtd_gp - prev_gp) / prev_gp) if prev_gp else None

    return {
        "mtd_sales": mtd_sales,
        "prev_sales": prev_sales,
        "mtd_orders": mtd_orders,
        "prev_orders": prev_orders,
        "mtd_gp": mtd_gp,
        "prev_gp": prev_gp,
        "sales_delta": sales_delta,
        "orders_delta": orders_delta,
        "gp_delta": gp_delta,
        "current_month_start": current_month_start.date(),
        "latest_date": latest_date.date(),
        "previous_month_start": previous_month_start.date(),
        "previous_same_day": previous_same_day.date(),
    }


def make_stock_summary(filtered_orders: pd.DataFrame, stock_df: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    if stock_df.empty:
        return pd.DataFrame()

    latest_order_date = filtered_orders["date_dt"].max()
    if pd.isna(latest_order_date):
        return pd.DataFrame()

    latest_order_date = pd.Timestamp(latest_order_date)
    lookback_start = latest_order_date - pd.Timedelta(days=lookback_days - 1)

    recent_orders = filtered_orders[
        (filtered_orders["date_dt"] >= lookback_start) & (filtered_orders["date_dt"] <= latest_order_date)
    ].copy()

    sales_lookback = (
        recent_orders.groupby(["partner_sku", "product_title", "brand"], dropna=False, as_index=False)
        .agg(orders_last_n_days=("orders", "sum"))
    )

    stock = stock_df.copy()
    stock["partner_sku"] = pd.to_numeric(stock["partner_sku"], errors="coerce").astype("Int64")
    stock["current_stock_qty"] = pd.to_numeric(stock["current_stock_qty"], errors="coerce").fillna(0)

    out = stock.merge(sales_lookback, on="partner_sku", how="left")
    out["product_title"] = out["product_title"].fillna(out.get("stock_title"))
    out["brand"] = out["brand"].fillna(out.get("stock_brand"))

    out["orders_last_n_days"] = out["orders_last_n_days"].fillna(0)
    out["avg_daily_orders"] = out["orders_last_n_days"] / lookback_days
    out["days_of_cover"] = out["current_stock_qty"] / out["avg_daily_orders"]
    out.loc[out["avg_daily_orders"] <= 0, "days_of_cover"] = pd.NA
    out["stock_risk"] = out.apply(
        lambda row: classify_stock_risk(row.get("days_of_cover"), row.get("current_stock_qty")),
        axis=1,
    )

    snapshot_col = pd.to_datetime(out["inventory_snapshot_at"], errors="coerce")
    out["inventory_snapshot_at"] = snapshot_col.dt.strftime("%Y-%m-%d %H:%M").fillna("")

    cols = [
        "partner_sku",
        "stock_sku",
        "product_title",
        "brand",
        "current_stock_qty",
        "stock_risk",
        "orders_last_n_days",
        "avg_daily_orders",
        "days_of_cover",
        "inventory_snapshot_at",
    ]
    existing_cols = [c for c in cols if c in out.columns]
    out = out[existing_cols].copy()
    return out.sort_values(["days_of_cover", "current_stock_qty"], ascending=[True, False], na_position="last")



def classify_stock_risk(days_of_cover, current_stock_qty):
    if pd.isna(current_stock_qty) or current_stock_qty <= 0:
        return "Out of stock"
    if pd.isna(days_of_cover):
        return "No recent sales"
    if days_of_cover < 14:
        return "Critical"
    if days_of_cover <= 30:
        return "Warning"
    return "Healthy"


def style_stock_coverage(df: pd.DataFrame):
    def row_style(row):
        risk = row.get("stock_risk")
    
    if risk == "Out of stock":
        color = "background-color: #f8d7da; color: black;"
    elif risk == "Critical":
        color = "background-color: #f8d7da; color: black;"
    elif risk == "Warning":
        color = "background-color: #fff3cd; color: black;"
    elif risk == "Healthy":
        color = "background-color: #d1e7dd; color: black;"
    else:
        color = ""
        return [color] * len(row)

    display = df.copy()
    for col in ["avg_daily_orders", "days_of_cover"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(1)

    return display.style.apply(row_style, axis=1)

def to_excel_bytes(data: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data["daily"].to_excel(writer, sheet_name="Daily Performance", index=False)
        data["monthly"].to_excel(writer, sheet_name="Monthly performance", index=False)
        data["product_orders"].to_excel(writer, sheet_name="Product Orders", index=False)
        data["product_sales"].to_excel(writer, sheet_name="Product Sales", index=False)
        data["product_metrics"].to_excel(writer, sheet_name="Product Metrics", index=False)
        data["top_gainers"].to_excel(writer, sheet_name="Top Gainers", index=False)
        data["top_decliners"].to_excel(writer, sheet_name="Top Decliners", index=False)
        data["stock_report"].to_excel(writer, sheet_name="Current Stock", index=False)
        data["stock_coverage"].to_excel(writer, sheet_name="Stock Coverage", index=False)
        data["missing_cogs"].to_excel(writer, sheet_name="Missing COGS", index=False)
        data["missing_fees"].to_excel(writer, sheet_name="Missing Fees", index=False)
        data["missing_fx"].to_excel(writer, sheet_name="Missing FX", index=False)
    return output.getvalue()


st.title("Noon Sales Dashboard")
st.caption("USD-only reporting from Noon orders, product reference, FX tables, and stock report.")

with st.sidebar:
    st.header("Files")
    st.write("Upload your 4 data files or keep local auto-load enabled.")
    orders_file = st.file_uploader("Orders file", type=["csv", "xlsx"], key="orders")
    ref_file = st.file_uploader("Product reference", type=["csv", "xlsx"], key="ref")
    fx_file = st.file_uploader("FX rates", type=["csv", "xlsx"], key="fx")
    stock_file = st.file_uploader("Stock report", type=["csv", "xlsx"], key="stock")
    use_sample = st.checkbox("Load local files automatically", value=True)
    coverage_days = st.number_input("Coverage lookback days", min_value=1, max_value=180, value=14, step=1)

if use_sample and not orders_file:
    orders_file = open_local("Noon Sales.csv")
if use_sample and not ref_file:
    ref_file = open_local("Product reference.csv")
if use_sample and not fx_file:
    fx_file = open_local("FX rates.csv")
if use_sample and not stock_file:
    stock_file = open_local("Inventory.csv")

if not all([orders_file, ref_file, fx_file]):
    st.info("Upload the orders, product reference, and FX files to start.")
    st.stop()

try:
    orders_df = read_orders(orders_file)
    ref_df = read_reference(ref_file)
    fx_df = read_fx(fx_file)
    stock_df = read_stock(stock_file) if stock_file else pd.DataFrame()
    model = build_model(orders_df, ref_df, fx_df)
except Exception as e:
    st.error(str(e))
    st.stop()

if model.empty:
    st.warning("No valid orders found after filtering statuses.")
    st.stop()

# Filters
brands = sorted([b for b in model["brand"].dropna().unique().tolist()])
currencies = sorted([c for c in model["currency"].dropna().unique().tolist()])
months = sorted(model["month"].dropna().unique().tolist())

min_date = model["date_dt"].min().date()
max_date = model["date_dt"].max().date()

f1, f2, f3, f4 = st.columns(4)

date_range = f1.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

selected_brands = f2.multiselect("Brand", brands, default=brands)
selected_currencies = f3.multiselect("Currency", currencies, default=currencies)
selected_months = f4.multiselect("Month", months, default=months)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

filtered = model[
    model["date_dt"].dt.date.between(start_date, end_date)
    & model["brand"].fillna("Unknown").isin(selected_brands or brands)
    & model["currency"].isin(selected_currencies or currencies)
    & model["month"].isin(selected_months or months)
].copy()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# KPI cards
total_orders = int(filtered["orders"].sum())
total_sales = float(filtered["sales_usd"].sum())
total_gp = float(filtered["gp_usd"].sum())
margin_pct = (total_gp / total_sales) if total_sales else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders", f"{total_orders:,}")
k2.metric("Sales USD", f"${total_sales:,.2f}")
k3.metric("GP USD", f"${total_gp:,.2f}")
k4.metric("Margin %", f"{margin_pct:.2%}")

# Prep tables
daily = make_daily_table(filtered)
monthly = make_monthly_table(filtered)
product_orders = make_product_orders(filtered)
product_sales = make_product_sales(filtered)
product_metrics = make_product_metrics(filtered)
top_gainers, top_decliners = make_gainers_decliners(filtered)
mtd = make_mtd_comparison(filtered)
stock_coverage = make_stock_summary(filtered, stock_df, lookback_days=coverage_days)
stock_report = stock_df.sort_values("current_stock_qty", ascending=False) if not stock_df.empty else pd.DataFrame()

missing_cogs = filtered[filtered["cogs_aed"].isna()][["partner_sku", "product_title", "brand"]].drop_duplicates()
missing_fees = filtered[filtered["fee_percent"].isna()][["partner_sku", "product_title", "brand"]].drop_duplicates()
missing_fx = filtered[filtered["to_usd"].isna()][["currency"]].drop_duplicates()

# Charts
st.subheader("Overview Charts")

chart1, chart2 = st.columns(2)

with chart1:
    sales_trend = (
        filtered.groupby("date", as_index=False)
        .agg(sales_usd=("sales_usd", "sum"))
        .sort_values("date")
    )
    st.line_chart(sales_trend.set_index("date"))

with chart2:
    orders_trend = (
        filtered.groupby("date", as_index=False)
        .agg(orders=("orders", "sum"))
        .sort_values("date")
    )
    st.bar_chart(orders_trend.set_index("date"))

chart3, chart4 = st.columns(2)

with chart3:
    gp_trend = (
        filtered.groupby("date", as_index=False)
        .agg(gp_usd=("gp_usd", "sum"))
        .sort_values("date")
    )
    st.line_chart(gp_trend.set_index("date"))

with chart4:
    brand_sales = (
        filtered.groupby("brand", dropna=False, as_index=False)
        .agg(sales_usd=("sales_usd", "sum"))
        .sort_values("sales_usd", ascending=False)
    )
    brand_sales["brand"] = brand_sales["brand"].fillna("Unknown")
    st.bar_chart(brand_sales.set_index("brand"))

tabs = st.tabs([
    "Daily Performance",
    "Monthly Performance",
    "Product Orders",
    "Product Sales",
    "Product Metrics",
    "Current Stock",
    "Stock Coverage",
    "Top Gainers / Decliners",
    "MTD vs Previous Month",
    "Data Quality",
])

with tabs[0]:
    st.dataframe(daily, use_container_width=True)

with tabs[1]:
    st.dataframe(monthly, use_container_width=True)

with tabs[2]:
    st.dataframe(product_orders, use_container_width=True)

with tabs[3]:
    st.dataframe(product_sales, use_container_width=True)

with tabs[4]:
    st.dataframe(product_metrics, use_container_width=True)

with tabs[5]:
    if stock_report.empty:
        st.info("Upload a stock report to see current stock by SKU.")
    else:
        st.dataframe(stock_report, use_container_width=True)

with tabs[6]:
    if stock_coverage.empty:
        st.info("Upload a stock report to see stock coverage days.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("SKUs in stock file", f"{stock_coverage['partner_sku'].nunique():,}")
        c2.metric("Critical < 14 days", f"{(stock_coverage['stock_risk'] == 'Critical').sum():,}")
        c3.metric("Warning 14-30 days", f"{(stock_coverage['stock_risk'] == 'Warning').sum():,}")
        c4.metric("Healthy > 30 days", f"{(stock_coverage['stock_risk'] == 'Healthy').sum():,}")
        c5.metric("Out of stock", f"{(stock_coverage['stock_risk'] == 'Out of stock').sum():,}")

        st.caption(
            f"Days of cover = current_stock_qty / (orders in last {coverage_days} days / {coverage_days}). "
            "Example: 140 orders in 14 days and stock 280 = 28.0 days of cover."
        )
        st.markdown("**Risk colors:** red = under 14 days or out of stock, yellow = 14 to 30 days, green = over 30 days.")
        st.dataframe(style_stock_coverage(stock_coverage), use_container_width=True)

with tabs[7]:
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Top Gainers")
        st.caption("Last 7 days vs previous 7 days by sales USD")
        st.dataframe(top_gainers, use_container_width=True)

    with g2:
        st.subheader("Top Decliners")
        st.caption("Last 7 days vs previous 7 days by sales USD")
        st.dataframe(top_decliners, use_container_width=True)

with tabs[8]:
    if mtd is None:
        st.info("No data available for MTD comparison.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "MTD Sales USD",
            f"${mtd['mtd_sales']:,.2f}",
            f"{mtd['sales_delta']:.2%}" if mtd["sales_delta"] is not None else "n/a",
        )
        m2.metric(
            "MTD Orders",
            f"{mtd['mtd_orders']:,}",
            f"{mtd['orders_delta']:.2%}" if mtd["orders_delta"] is not None else "n/a",
        )
        m3.metric(
            "MTD GP USD",
            f"${mtd['mtd_gp']:,.2f}",
            f"{mtd['gp_delta']:.2%}" if mtd["gp_delta"] is not None else "n/a",
        )

        st.caption(
            f"Current MTD: {mtd['current_month_start']} to {mtd['latest_date']} | "
            f"Previous comparison: {mtd['previous_month_start']} to {mtd['previous_same_day']}"
        )

with tabs[9]:
    d1, d2, d3 = st.columns(3)
    d1.metric("Missing COGS SKUs", f"{missing_cogs['partner_sku'].nunique() if not missing_cogs.empty else 0}")
    d2.metric("Missing Fee SKUs", f"{missing_fees['partner_sku'].nunique() if not missing_fees.empty else 0}")
    d3.metric("Missing FX Currencies", f"{missing_fx['currency'].nunique() if not missing_fx.empty else 0}")

    if not missing_cogs.empty or not missing_fees.empty or not missing_fx.empty:
        st.warning("Some SKUs or currencies are missing setup data.")
    else:
        st.success("No missing COGS, fee, or FX issues found.")

    if not missing_cogs.empty:
        st.subheader("Missing COGS")
        st.dataframe(missing_cogs, use_container_width=True)

    if not missing_fees.empty:
        st.subheader("Missing Fees")
        st.dataframe(missing_fees, use_container_width=True)

    if not missing_fx.empty:
        st.subheader("Missing FX")
        st.dataframe(missing_fx, use_container_width=True)

excel_bytes = to_excel_bytes({
    "daily": daily,
    "monthly": monthly,
    "product_orders": product_orders,
    "product_sales": product_sales,
    "product_metrics": product_metrics,
    "top_gainers": top_gainers,
    "top_decliners": top_decliners,
    "stock_report": stock_report,
    "stock_coverage": stock_coverage,
    "missing_cogs": missing_cogs,
    "missing_fees": missing_fees,
    "missing_fx": missing_fx,
})

st.download_button(
    "Download report as Excel",
    data=excel_bytes,
    file_name="noon_sales_report_usd.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
