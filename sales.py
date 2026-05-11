# -------------------------------------------------------------
# 🛒 Supermarket Sales Analysis & Forecasting Dashboard
# - Advanced but friendly UI
# - Dataset overview at top
# - Simple forecasting
# - Theme toggle (Light/Dark)
# - Green labels for Category & City filters
# - Python 3.12 compatible
# -------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False


# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Supermarket Sales Dashboard",
    page_icon="🛒",
    layout="wide",
)


# ------------------------------------------------------------- 
# THEME HANDLER
# -------------------------------------------------------------
def apply_theme(theme: str):
    """Inject CSS based on Light/Dark theme."""
    if theme == "Dark":
        bg_main = "#020617"
        text_color = "#e5e7eb"
        card_bg = "rgba(15,23,42,0.95)"
        border_color = "rgba(148,163,184,0.4)"
    else:  # Light
        bg_main = "#f3f4f6"
        text_color = "#111827"
        card_bg = "#ffffff"
        border_color = "rgba(209,213,219,0.9)"

    st.markdown(
        f"""
        <style>
        body {{
            background: {bg_main};
        }}
        .main {{
            color: {text_color};
            background: {bg_main};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}
        .section-card {{
            padding: 1.2rem 1.4rem;
            border-radius: 1rem;
            background: {card_bg};
            border: 1px solid {border_color};
            margin-bottom: 1rem;
        }}
        .metric-card {{
            padding: 0.9rem 1rem;
            border-radius: 0.9rem;
            background: rgba(0,0,0,0.1);
            border: 1px solid rgba(148,163,184,0.5);
        }}
        .green-label {{
            color: #16a34a !important; /* green-600 */
            font-weight: 600;
            margin-bottom: 0.2rem;
        }}
        .stDownloadButton button {{
            border-radius: 999px !important;
            padding: 0.4rem 1.2rem !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# DATA LOADING & CLEANING
# -------------------------------------------------------------
def load_data(uploaded):
    """Read CSV/Excel into DataFrame."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded)
        st.error("❌ Unsupported file format. Please upload CSV or Excel.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return pd.DataFrame()


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df


def detect_columns(df: pd.DataFrame) -> dict:
    """Automatically detect important columns."""
    def find(patterns):
        for col in df.columns:
            for p in patterns:
                if p in col:
                    return col
        return None

    date_col = find(["date", "order_date", "invoice_date", "bill_date"])
    sales_col = find(["total_sales", "sales", "amount", "revenue", "grand_total"])
    category_col = find(["category", "product", "item", "sku", "description"])

    optional_cols = {
        "gender": find(["gender", "sex"]),
        "city": find(["city", "branch", "location", "store"]),
        "discount": find(["discount", "offer"]),
        "rating": find(["rating", "review"]),
        "age": find(["age"]),
        "payment_method": find(["payment", "pay_mode"]),
        "profit": find(["profit", "margin"]),
    }


    return {
        "date": date_col,
        "sales": sales_col,
        "category": category_col,
        "optional": optional_cols,
    }


def compute_sales_if_missing(df: pd.DataFrame, col_info: dict):
    """Compute total_sales = quantity × unit_price if no sales column."""
    df = df.copy()
    if col_info["sales"] is not None:
        return df, col_info

    qty_candidates = ["quantity", "qty", "qty_sold", "units"]
    price_candidates = ["unit_price", "price", "rate", "selling_price"]

    qty_col = None
    price_col = None

    for c in df.columns:
        for q in qty_candidates:
            if q in c:
                qty_col = c
        for p in price_candidates:
            if p in c:
                price_col = c

    if qty_col and price_col:
        df["total_sales"] = (
            pd.to_numeric(df[qty_col], errors="coerce")
            * pd.to_numeric(df[price_col], errors="coerce")
        )
        col_info["sales"] = "total_sales"
        st.info(f"ℹ️ No explicit sales column found. Created **total_sales = {qty_col} × {price_col}**.")
    else:
        st.warning("⚠️ No sales column and could not compute from quantity × price.")

    return df, col_info


def clean_data(df: pd.DataFrame):
    df = standardize(df)
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)

    col_info = detect_columns(df)
    df, col_info = compute_sales_if_missing(df, col_info)

    date_col = col_info["date"]
    sales_col = col_info["sales"]

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ {invalid_dates} rows have invalid dates and were dropped.")
        df.dropna(subset=[date_col], inplace=True)
    else:
        st.error("❌ No date-like column detected. Forecasting will be disabled.")

    if sales_col:
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
        invalid_sales = df[sales_col].isna().sum()
        if invalid_sales > 0:
            st.warning(f"⚠️ {invalid_sales} rows have invalid sales and were dropped.")
        df.dropna(subset=[sales_col], inplace=True)
    else:
        st.error("❌ No sales column detected. Some analyses will be limited.")

    return df, col_info


# -------------------------------------------------------------
# KPI ANIMATION
# -------------------------------------------------------------
def animate_metric(label: str, value: float, fmt: str = "{:,.2f}", duration: float = 0.6):
    placeholder = st.empty()
    steps = 25
    if steps <= 0:
        steps = 1
    delay = duration / steps

    for i in range(steps):
        current = value * (i + 1) / steps
        placeholder.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;opacity:0.8;">{label}</div>
                <div style="font-size:1.6rem;font-weight:700;margin-top:0.1rem;">
                    {fmt.format(current)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(delay)


# -------------------------------------------------------------
# SEGMENT SUMMARY (Gender/City/Payment...)
# -------------------------------------------------------------
def segment_summary(df: pd.DataFrame, segment_col: str, sales_col: str, quantity_col: str = None):
    df = df.copy()

    if quantity_col is None:
        df["_temp_qty"] = 1
        quantity_col = "_temp_qty"

    summary = df.groupby(segment_col).agg(
        total_quantity=(quantity_col, "sum"),
        total_sales=(sales_col, "sum"),
        transactions=(quantity_col, "count"),
    ).reset_index()

    return summary


# -------------------------------------------------------------
# EDA CHART HELPERS
# -------------------------------------------------------------
def category_sales_chart(df, category_col, sales_col):
    agg = df.groupby(category_col)[sales_col].sum().reset_index()
    agg = agg.sort_values(sales_col, ascending=False)
    fig = px.bar(
        agg,
        x=category_col,
        y=sales_col,
        title="🧺 Category-wise Sales",
    )
    st.plotly_chart(fig, use_container_width=True)


def profit_analysis(df, col_info):
    profit_col = col_info["optional"].get("profit")
    category_col = col_info["category"]
    sales_col = col_info["sales"]

    if not (profit_col and category_col and sales_col):
        st.info("ℹ️ Profit analysis requires profit, category and sales columns.")
        return

    agg = (
        df.groupby(category_col)
        .agg(total_sales=(sales_col, "sum"), total_profit=(profit_col, "sum"))
        .reset_index()
    )
    agg["profit_margin_%"] = (agg["total_profit"] / agg["total_sales"]) * 100

    fig = px.bar(
        agg.sort_values("total_profit", ascending=False),
        x=category_col,
        y="total_profit",
        title="💹 Profit by Category",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("📋 Profit Table")
    st.dataframe(agg.sort_values("profit_margin_%", ascending=False))


def correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        st.info("ℹ️ Not enough numeric columns for correlation heatmap.")
        return
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="🔥 Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)


def top_products_chart(df, col_info, top_n=10):
    category_col = col_info["category"]
    sales_col = col_info["sales"]
    if not (category_col and sales_col):
        return

    agg = df.groupby(category_col)[sales_col].sum().reset_index()
    agg = agg.sort_values(sales_col, ascending=False).head(top_n)
    fig = px.bar(
        agg,
        x=category_col,
        y=sales_col,
        title=f"🏆 Top {top_n} Best-Selling Categories/Products",
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# SIMPLE FORECASTING (Non-technical friendly)
# -------------------------------------------------------------
def simple_forecast(df, col_info, periods):
    date_col = col_info["date"]
    sales_col = col_info["sales"]

    if not (date_col and sales_col):
        st.error("❌ Forecasting requires both date and sales columns.")
        return None, None

    ts = df.groupby(date_col)[sales_col].sum().reset_index()
    ts = ts.rename(columns={date_col: "ds", sales_col: "y"})

    if len(ts) < 5:
        st.warning("⚠️ Not enough data points for meaningful forecasting.")
        return None, None

    # Prophet preferred
    if PROPHET_AVAILABLE:
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)[["ds", "yhat"]]
    elif ARIMA_AVAILABLE:
        series = ts.set_index("ds")["y"]
        model = ARIMA(series, order=(1, 1, 1)).fit()
        fc_res = model.get_forecast(periods)
        forecast = fc_res.summary_frame()[["mean"]].rename(columns={"mean": "yhat"})
        forecast["ds"] = forecast.index
        forecast = forecast[["ds", "yhat"]]
    else:
        st.error("❌ No forecasting library found (install prophet or statsmodels).")
        return None, None

    # Build simple Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["ds"],
            y=ts["y"],
            mode="lines",
            name="Past Sales",
            line=dict(color="skyblue", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="lightgreen", width=3),
        )
    )
    fig.update_layout(
        title=f"📈 Sales Forecast (Next {periods} Days)",
        xaxis_title="Date",
        yaxis_title="Sales",
    )

    return forecast, fig


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
def main():
    # Sidebar: theme first so CSS applies before layout
    st.sidebar.header("🎨 Theme & Settings")
    theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
    apply_theme(theme)

    st.markdown(
        "<h1 style='text-align:center;'>Supermarket Sales Analysis & Forecasting</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;opacity:0.8;'>Automatic EDA, customer insights, and sales forecasting.</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    forecast_period = st.sidebar.slider("Forecast days", 7, 90, 30)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segment for Comparison")
    segment_choice = st.sidebar.selectbox(
        "Choose segment",
        ["gender", "city", "payment_method", "age", "rating", "discount"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ Use a supermarket-style dataset with date, sales, and product/category columns. "
        "Optional columns like gender, city, payment_method make the analysis richer."
    )

    if uploaded is None:
        st.warning("📂 Please upload a dataset to start.")
        return

    raw_df = load_data(uploaded)
    if raw_df.empty:
        st.error("❌ Could not load the dataset.")
        return

    df, col_info = clean_data(raw_df)
    if df.empty:
        st.error("❌ No valid rows after cleaning. Please check your data.")
        return

    date_col = col_info["date"]
    sales_col = col_info["sales"]
    category_col = col_info["category"]
    optional_cols = col_info["optional"]

    # ---------------------------------------------------------
    # SECTION 1: DATASET OVERVIEW (TOP)
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Dataset Overview")

    if date_col and sales_col:
        total_revenue = float(df[sales_col].sum())
        total_txn = int(len(df))
        avg_daily = float(df.groupby(date_col)[sales_col].sum().mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            animate_metric("Total Revenue", total_revenue, "{:,.2f}")
        with c2:
            animate_metric("Total Transactions", total_txn, "{:,.0f}")
        with c3:
            animate_metric("Avg Daily Sales", avg_daily, "{:,.2f}")

    with st.expander("View Sample Data"):
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: FILTERS (with green labels)
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Filters")

    df_filtered = df.copy()

    # Date range filter (safe for slider)
    if date_col:
        min_ts = df[date_col].min()
        max_ts = df[date_col].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            min_date = min_ts.date()
            max_date = max_ts.date()
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
            )
            df_filtered = df_filtered[
                (df_filtered[date_col].dt.date >= date_range[0])
                & (df_filtered[date_col].dt.date <= date_range[1])
            ]

    # Category filter
    if category_col:
        st.markdown("<p class='green-label'>Filter by Category</p>", unsafe_allow_html=True)
        cats = sorted(df[category_col].dropna().unique().tolist())
        selected_cats = st.multiselect("", cats, default=cats)
        if selected_cats:
            df_filtered = df_filtered[df_filtered[category_col].isin(selected_cats)]

    # City filter
    city_col = optional_cols.get("city")
    if city_col:
        st.markdown("<p class='green-label'>Filter by City</p>", unsafe_allow_html=True)
        cities = sorted(df[city_col].dropna().unique().tolist())
        selected_cities = st.multiselect(" ", cities, default=cities)
        if selected_cities:
            df_filtered = df_filtered[df_filtered[city_col].isin(selected_cities)]

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 3: AUTOMATIC EDA
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Automatic EDA")

    if category_col and sales_col:
        category_sales_chart(df_filtered, category_col, sales_col)

    top_products_chart(df_filtered, col_info, top_n=10)
    correlation_heatmap(df_filtered)
    profit_analysis(df_filtered, col_info)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 4: CUSTOMER SEGMENT COMPARISON
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Customer Segment Comparison")

    seg_col = optional_cols.get(segment_choice)
    if seg_col and sales_col:
        # Detect quantity column
        quantity_col = None
        for q in ["quantity", "qty", "units"]:
            for c in df_filtered.columns:
                if q in c:
                    quantity_col = c
                    break
            if quantity_col:
                break

        summary_df = segment_summary(df_filtered, seg_col, sales_col, quantity_col)
        st.write("Summary Table")
        st.dataframe(summary_df)

        fig_seg = px.bar(
            summary_df,
            x=seg_col,
            y="total_sales",
            text_auto=True,
            color=seg_col,
            title=f"Total Sales by {segment_choice.capitalize()}",
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info(f"ℹ️ Column '{segment_choice}' not found in this dataset.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 5: SIMPLE FORECASTING DASHBOARD
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Sales Forecasting")

    st.write(
        """
        This forecast shows how much you may sell in the coming days, based on past trends.
        
        - **Blue line** → Past sales  
        - **Green line** → Forecasted sales  
        """
    )

    forecast_df, forecast_fig = simple_forecast(df_filtered, col_info, forecast_period)

    if forecast_fig is not None and forecast_df is not None:
        st.plotly_chart(forecast_fig, use_container_width=True)

        st.subheader("Forecast Summary")
        future_part = forecast_df.tail(forecast_period)

        total_future = future_part["yhat"].sum()
        best_row = future_part.loc[future_part["yhat"].idxmax()]

        st.write(f"📌 **Total predicted sales (next {forecast_period} days):** `{total_future:,.2f}`")
        st.write(
            f"📌 **Peak predicted day:** `{best_row['ds'].date()}` "
            f"with expected sales of `{best_row['yhat']:.2f}`"
        )

        st.dataframe(future_part.rename(columns={"ds": "Date", "yhat": "Predicted Sales"}))
    else:
        st.info("ℹ️ Forecast not available for this dataset.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 6: DOWNLOADS
    # ---------------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("⬇ Downloads")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Cleaned Dataset",
            df.to_csv(index=False),
            "cleaned_dataset.csv",
            mime="text/csv",
        )
    with c2:
        if forecast_df is not None:
            st.download_button(
                "Download Forecast Results",
                forecast_df.to_csv(index=False),
                "forecast_results.csv",
                mime="text/csv",
            )
        else:
            st.info("Forecast CSV will be available once a forecast is generated.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# RUN APP
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
