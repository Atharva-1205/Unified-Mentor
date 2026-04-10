import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Profit Optimization Intelligence System",
    layout="wide"
)

st.title("📊 Profit Optimization Intelligence System")
st.markdown("Comprehensive Product Profitability & Margin Intelligence")

# -------------------------------------------------------
# LOAD DATA (TRANSACTION LEVEL)
# -------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Nassau Candy Distributor.xlsx")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# -------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------
st.sidebar.header("Filters & Controls")

min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

start_date, end_date = date_range

selected_division = st.sidebar.selectbox(
    "Select Division",
    ["All"] + sorted(df["Division"].unique())
)

margin_threshold = st.sidebar.slider(
    "Margin Risk Threshold",
    0.0,
    1.0,
    0.15
)

product_search = st.sidebar.text_input("Search Product")

# -------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------
filtered_df = df[
    (df["Order Date"] >= pd.to_datetime(start_date)) &
    (df["Order Date"] <= pd.to_datetime(end_date))
]

if selected_division != "All":
    filtered_df = filtered_df[
        filtered_df["Division"] == selected_division
    ]

if product_search:
    filtered_df = filtered_df[
        filtered_df["Product Name"].str.contains(product_search, case=False)
    ]

# -------------------------------------------------------
# PRODUCT-LEVEL AGGREGATION
# -------------------------------------------------------
product_summary = (
    filtered_df.groupby(
        ["Product ID", "Product Name", "Division"]
    )
    .agg({
        "Sales": "sum",
        "Cost": "sum",
        "Gross Profit": "sum",
        "Units": "sum"
    })
    .reset_index()
)

# KPIs
product_summary["Gross Margin %"] = (
    product_summary["Gross Profit"] /
    product_summary["Sales"]
)

product_summary["Profit per Unit"] = (
    product_summary["Gross Profit"] /
    product_summary["Units"]
)

product_summary["Revenue Contribution"] = (
    product_summary["Sales"] /
    product_summary["Sales"].sum()
)

product_summary["Profit Contribution"] = (
    product_summary["Gross Profit"] /
    product_summary["Gross Profit"].sum()
)

# Margin Volatility (Std Dev of monthly margin)
monthly_margin = (
    filtered_df
    .assign(Month=filtered_df["Order Date"].dt.to_period("M"))
    .groupby(["Product ID", "Month"])
    .agg({
        "Sales": "sum",
        "Gross Profit": "sum"
    })
    .reset_index()
)

monthly_margin["Monthly Margin"] = (
    monthly_margin["Gross Profit"] /
    monthly_margin["Sales"]
)

margin_volatility = (
    monthly_margin.groupby("Product ID")["Monthly Margin"]
    .std()
    .reset_index()
    .rename(columns={"Monthly Margin": "Margin Volatility"})
)

product_summary = product_summary.merge(
    margin_volatility,
    on="Product ID",
    how="left"
)

# -------------------------------------------------------
# KPI SECTION (ALIGNED TO REQUIREMENT TABLE)
# -------------------------------------------------------

# Company-level calculations
total_sales = product_summary["Sales"].sum()
total_profit = product_summary["Gross Profit"].sum()

gross_margin = total_profit / total_sales if total_sales > 0 else 0

profit_per_unit = (
    total_profit / product_summary["Units"].sum()
    if product_summary["Units"].sum() > 0 else 0
)

revenue_contribution = 1.0   # company level = 100%
profit_contribution = 1.0    # company level = 100%

margin_volatility = product_summary["Margin Volatility"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Gross Margin (%)", f"{gross_margin:.1%}")
col2.metric("Profit per Unit", f"${profit_per_unit:.2f}")
col3.metric("Revenue Contribution", f"{revenue_contribution:.0%}")
col4.metric("Profit Contribution", f"{profit_contribution:.0%}")
col5.metric("Margin Volatility", f"{margin_volatility:.6f}")

# ======================================================
# PRODUCT PROFITABILITY OVERVIEW
# ======================================================
st.header("Product Profitability Overview")

st.subheader("Product-Level Margin Leaderboard")
st.dataframe(
    product_summary.sort_values(
        "Gross Margin %", ascending=False
    ).head(20)
)

st.subheader("Top Products by Profit Contribution")

fig_profit_contrib = px.bar(
    product_summary.sort_values(
        "Profit Contribution", ascending=False
    ).head(15),
    x="Product Name",
    y="Profit Contribution",
    title="Top Profit Contributing Products"
)

st.plotly_chart(fig_profit_contrib, width="stretch")

st.divider()

# ======================================================
# DIVISION PERFORMANCE DASHBOARD
# ======================================================
st.header("Division Performance Dashboard")

division_summary = (
    product_summary.groupby("Division")
    .agg({
        "Sales": "sum",
        "Gross Profit": "sum"
    })
    .reset_index()
)

division_summary["Gross Margin %"] = (
    division_summary["Gross Profit"] /
    division_summary["Sales"]
)

# Revenue vs Profit
fig_rev_profit = px.bar(
    division_summary,
    x="Division",
    y=["Sales", "Gross Profit"],
    barmode="group",
    title="Revenue vs Profit by Division"
)

st.plotly_chart(fig_rev_profit, width="stretch")

# Margin Distribution
fig_margin_dist = px.box(
    product_summary,
    x="Division",
    y="Gross Margin %",
    title="Margin Distribution by Division"
)

st.plotly_chart(fig_margin_dist, width="stretch")

st.divider()

# ======================================================
# COST VS MARGIN DIAGNOSTICS
# ======================================================
st.header("Cost vs Margin Diagnostics")

# Cost-Sales Scatter
fig_cost_sales = px.scatter(
    product_summary,
    x="Cost",
    y="Sales",
    size="Units",
    color="Division",
    hover_name="Product Name",
    title="Cost vs Sales Scatter Plot"
)

st.plotly_chart(fig_cost_sales, width="stretch")

# Margin Risk Flags
risk_products = product_summary[
    product_summary["Gross Margin %"] < margin_threshold
]

st.metric("Products Below Margin Threshold",
          len(risk_products))

st.dataframe(risk_products)

st.divider()

# ======================================================
# PROFIT CONCENTRATION ANALYSIS (UPGRADED)
# ======================================================
st.header("Profit Concentration Analysis")

pareto_df = product_summary.sort_values(
    "Gross Profit", ascending=False
).reset_index(drop=True)

pareto_df["Cumulative Profit"] = pareto_df["Gross Profit"].cumsum()
pareto_df["Cumulative %"] = (
    pareto_df["Cumulative Profit"] /
    pareto_df["Gross Profit"].sum()
)

pareto_df["Rank"] = pareto_df.index + 1

# Identify products driving 80%
products_80 = pareto_df[
    pareto_df["Cumulative %"] <= 0.8
]

fig_pareto = px.line(
    pareto_df,
    x="Rank",
    y="Cumulative %",
    title="Pareto Curve (Profit Concentration)"
)

# 80% horizontal line
fig_pareto.add_hline(y=0.8, line_dash="dash")

# Highlight 80% drivers
fig_pareto.add_scatter(
    x=products_80["Rank"],
    y=products_80["Cumulative %"],
    mode="markers+text",
    text=products_80["Product Name"],
    textposition="top center",
    marker=dict(size=10),
    name="80% Profit Drivers"
)

st.plotly_chart(fig_pareto, width="stretch")

st.metric(
    "Products Driving 80% of Profit",
    f"{len(products_80)} ({len(products_80)/len(pareto_df):.1%})"
)

st.subheader("Products Driving 80% of Profit")

st.dataframe(
    products_80[[
        "Product Name",
        "Division",
        "Sales",
        "Gross Profit",
        "Gross Margin %"
    ]]
)