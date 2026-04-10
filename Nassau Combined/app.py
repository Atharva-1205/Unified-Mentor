import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


def apply_unified_theme():
    st.markdown("""
<style>
:root {
    --ui-text: #1f2937;
    --ui-muted: #475569;
    --ui-panel: #eef3f7;
    --ui-border: #d3dde8;
    --ui-accent: #0f766e;
    --ui-accent-soft: #dceeea;
    --ui-accent-ink: #134e4a;
}

.main .block-container {
    padding-top: 1.2rem;
}

h1, h2, h3 {
    color: var(--ui-text);
}

div[data-testid="stMetric"] {
    background: var(--ui-panel);
    border: 1px solid var(--ui-border);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
}

div[data-testid="stMetricLabel"] {
    color: var(--ui-muted);
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ui-text) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--ui-muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: var(--ui-accent-ink) !important;
}

span[data-baseweb="tag"] {
    background-color: var(--ui-accent-soft) !important;
    color: var(--ui-accent-ink) !important;
    border-radius: 6px !important;
}

/* streamlit-option-menu */
ul.nav {
    gap: 0.3rem;
}

ul.nav .nav-link {
    color: var(--ui-text) !important;
    border-radius: 10px !important;
}

ul.nav .nav-link svg {
    color: var(--ui-accent-ink) !important;
}

ul.nav .nav-link.active {
    background-color: var(--ui-accent-soft) !important;
    color: var(--ui-accent-ink) !important;
}

ul.nav .nav-link.active svg {
    color: var(--ui-accent-ink) !important;
}
</style>
""", unsafe_allow_html=True)


def render_page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def render_profit_intelligence():
    render_page_header(
        "Profit Optimization Intelligence System",
        "Comprehensive Product Profitability and Margin Intelligence"
    )

    @st.cache_data
    def load_data():
        df = pd.read_excel("Nassau Candy Distributor.xlsx")
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        return df

    df = load_data()

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

    total_sales = product_summary["Sales"].sum()
    total_profit = product_summary["Gross Profit"].sum()

    gross_margin = total_profit / total_sales if total_sales > 0 else 0

    profit_per_unit = (
        total_profit / product_summary["Units"].sum()
        if product_summary["Units"].sum() > 0 else 0
    )

    revenue_contribution = 1.0
    profit_contribution = 1.0

    margin_volatility = product_summary["Margin Volatility"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Gross Margin (%)", f"{gross_margin:.1%}")
    col2.metric("Profit per Unit", f"${profit_per_unit:.2f}")
    col3.metric("Revenue Contribution", f"{revenue_contribution:.0%}")
    col4.metric("Profit Contribution", f"{profit_contribution:.0%}")
    col5.metric("Margin Volatility", f"{margin_volatility:.6f}")

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

    fig_rev_profit = px.bar(
        division_summary,
        x="Division",
        y=["Sales", "Gross Profit"],
        barmode="group",
        title="Revenue vs Profit by Division"
    )

    st.plotly_chart(fig_rev_profit, width="stretch")

    fig_margin_dist = px.box(
        product_summary,
        x="Division",
        y="Gross Margin %",
        title="Margin Distribution by Division"
    )

    st.plotly_chart(fig_margin_dist, width="stretch")

    st.divider()

    st.header("Cost vs Margin Diagnostics")

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

    risk_products = product_summary[
        product_summary["Gross Margin %"] < margin_threshold
    ]

    st.metric("Products Below Margin Threshold",
              len(risk_products))

    st.dataframe(risk_products)

    st.divider()

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

    products_80 = pareto_df[
        pareto_df["Cumulative %"] <= 0.8
    ]

    fig_pareto = px.line(
        pareto_df,
        x="Rank",
        y="Cumulative %",
        title="Pareto Curve (Profit Concentration)"
    )

    fig_pareto.add_hline(y=0.8, line_dash="dash")

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


def render_shipping_intelligence():
    @st.cache_data
    def load_data():
        df = pd.read_excel("cleaned_nassau_shipping_data.xlsx")

        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Ship Date"] = pd.to_datetime(df["Ship Date"])

        df["Shipping Lead Time"] = (
            df["Ship Date"] - df["Order Date"]
        ).dt.days

        return df

    df = load_data()

    us_state_codes = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY"
    }

    df["State Code"] = df["State/Province"].map(us_state_codes)

    st.sidebar.header("Filters & Controls")

    date_range = st.sidebar.date_input(
        "Date Range",
        [df["Order Date"].min(), df["Order Date"].max()]
    )

    region = st.sidebar.multiselect(
        "Region",
        sorted(df["Region"].unique())
    )

    state = st.sidebar.multiselect(
        "State",
        sorted(df["State/Province"].unique())
    )

    ship_mode = st.sidebar.multiselect(
        "Ship Mode",
        sorted(df["Ship Mode"].unique())
    )

    delay_threshold = st.sidebar.slider(
        "Delay Threshold (Days)",
        1, 15, 5
    )

    filtered_df = df.copy()

    if region:
        filtered_df = filtered_df[filtered_df["Region"].isin(region)]

    if state:
        filtered_df = filtered_df[filtered_df["State/Province"].isin(state)]

    if ship_mode:
        filtered_df = filtered_df[filtered_df["Ship Mode"].isin(ship_mode)]

    filtered_df = filtered_df[
        (filtered_df["Order Date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Order Date"] <= pd.to_datetime(date_range[1]))
    ]

    render_page_header(
        "Nassau Candy Distributor Shipping Intelligence Dashboard",
        "Shipping performance, delays, and route efficiency insights"
    )

    st.divider()

    avg_lead_time = filtered_df["Shipping Lead Time"].mean()

    route_volume = filtered_df.shape[0]

    delay_frequency = (
        (filtered_df["Shipping Lead Time"] > delay_threshold).mean() * 100
    )

    route_perf = filtered_df.groupby("Route").agg(
        avg_lead_time=("Shipping Lead Time", "mean"),
        shipments=("Order ID", "count")
    ).reset_index()

    route_perf["Efficiency Score"] = 1 - (
        (route_perf["avg_lead_time"] - route_perf["avg_lead_time"].min()) /
        (route_perf["avg_lead_time"].max() - route_perf["avg_lead_time"].min())
    )

    efficiency_score = route_perf["Efficiency Score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Shipping Lead Time", f"{avg_lead_time:.1f}")
    c2.metric("Average Lead Time", f"{route_perf['avg_lead_time'].mean():.1f}")
    c3.metric("Route Volume", f"{route_volume}")
    c4.metric("Delay Frequency", f"{delay_frequency:.1f}%")
    c5.metric("Route Efficiency Score", f"{efficiency_score:.2f}")

    st.divider()

    st.subheader("Route Efficiency Overview")

    top_routes = route_perf.sort_values("avg_lead_time").head(10)
    slow_routes = route_perf.sort_values(
        "avg_lead_time", ascending=False
    ).head(10)

    col1, col2 = st.columns(2)

    fig1 = px.bar(
        top_routes,
        x="avg_lead_time",
        y="Route",
        orientation="h",
        title="Top 10 Fastest Routes"
    )

    fig2 = px.bar(
        slow_routes,
        x="avg_lead_time",
        y="Route",
        orientation="h",
        title="Top 10 Slowest Routes"
    )

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("US Shipping Efficiency Map")

    state_perf = filtered_df.groupby("State Code").agg(
        avg_lead_time=("Shipping Lead Time", "mean")
    ).reset_index()

    fig_map = px.choropleth(
        state_perf,
        locations="State Code",
        locationmode="USA-states",
        color="avg_lead_time",
        scope="usa"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    st.subheader("Ship Mode Performance")

    ship_perf = filtered_df.groupby("Ship Mode").agg(
        avg_lead_time=("Shipping Lead Time", "mean"),
        shipments=("Order ID", "count")
    ).reset_index()

    fig_ship = px.bar(
        ship_perf,
        x="Ship Mode",
        y="avg_lead_time",
        text="avg_lead_time"
    )

    st.plotly_chart(fig_ship, use_container_width=True)

    st.divider()

    st.subheader("Shipping Lead Time Distribution")

    fig_hist = px.histogram(
        filtered_df,
        x="Shipping Lead Time",
        nbins=40
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    st.subheader("Route Volume vs Lead Time")

    fig_scatter = px.scatter(
        route_perf,
        x="shipments",
        y="avg_lead_time",
        size="shipments",
        hover_data=["Route"]
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


def render_factory_optimization():
    df = pd.read_excel("Nassau Candy Distributor.xlsx")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin"] = df["Gross Profit"] / df["Sales"]

    model = pickle.load(open("leadtime_model.pkl", "rb"))
    le_region = pickle.load(open("le_region.pkl", "rb"))
    le_ship = pickle.load(open("le_ship.pkl", "rb"))
    le_product = pickle.load(open("le_product.pkl", "rb"))
    le_factory = pickle.load(open("le_factory.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    factories = list(le_factory.classes_)

    st.sidebar.header("Filters & Controls")

    product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
    region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
    shipmode = st.sidebar.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))

    priority = st.sidebar.slider(
        "Optimization Priority (Speed ←→ Profit)",
        0, 100, 50
    )

    def predict_lead_time(product, region, shipmode, factory):
        product_enc = le_product.transform([product])[0]
        region_enc = le_region.transform([region])[0]
        ship_enc = le_ship.transform([shipmode])[0]
        factory_enc = le_factory.transform([factory])[0]

        avg_vals = df[["Sales", "Units", "Cost"]].mean().values
        avg_scaled = scaler.transform([avg_vals])[0]

        row = np.array([[
            region_enc,
            ship_enc,
            product_enc,
            factory_enc,
            avg_scaled[0],
            avg_scaled[1],
            avg_scaled[2]
        ]])

        return model.predict(row)[0]

    def simulate():
        base_margin = df[df["Product Name"] == product]["Profit Margin"].mean()
        results = []

        for i, f in enumerate(factories):
            lead_time = predict_lead_time(product, region, shipmode, f)

            lead_time = lead_time * (1 + (i * 0.02))

            profit_adj = base_margin - (lead_time * 0.003)

            results.append([f, lead_time, profit_adj])

        return pd.DataFrame(
            results,
            columns=["Factory", "Predicted Lead Time", "Profit Impact"]
        )

    sim = simulate()

    current = df[
        (df["Product Name"] == product) &
        (df["Region"] == region)
    ]["Lead Time"].mean()

    if np.isnan(current):
        current = df["Lead Time"].mean()

    best = sim["Predicted Lead Time"].min()

    lead_reduction = ((current - best) / current) * 100
    profit_stability = sim["Profit Impact"].std()
    confidence_score = 0.85

    better_options = sim[sim["Predicted Lead Time"] < current]
    coverage = len(better_options) / len(sim)

    lt_range = sim["Predicted Lead Time"].max() - sim["Predicted Lead Time"].min()
    profit_range = sim["Profit Impact"].max() - sim["Profit Impact"].min()

    lt_norm = (sim["Predicted Lead Time"] - sim["Predicted Lead Time"].min()) / (lt_range + 1e-6)
    profit_norm = (sim["Profit Impact"] - sim["Profit Impact"].min()) / (profit_range + 1e-6)

    weight = priority / 100

    sim["Score"] = (
        (1 - weight) * lt_norm +
        weight * (1 - profit_norm)
    )

    sim_sorted = sim.sort_values("Score")

    render_page_header(
        "Factory Optimization Simulator",
        "Lead-time optimization and factory recommendation intelligence"
    )

    st.subheader("Key Performance Indicators")

    k1, k2, k3, k4 = st.columns(4)

    if lead_reduction > 0:
        k1.metric("Lead Time Reduction %", f"{round(lead_reduction,2)}%", delta="Improvement", delta_color="normal")
    elif lead_reduction < 0:
        k1.metric("Lead Time Reduction %", f"{round(lead_reduction,2)}%", delta="Worse", delta_color="inverse")
    else:
        k1.metric("Lead Time Reduction %", "0%", delta="No change", delta_color="off")

    k2.metric("Profit Stability", round(profit_stability, 4))
    k3.metric("Confidence Score", confidence_score)
    k4.metric("Recommendation Coverage", round(coverage * 100, 2))

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(sim_sorted, use_container_width=True)

    with col2:
        st.metric("Best Factory", sim_sorted.iloc[0]["Factory"])
        st.metric("Best Lead Time", round(sim_sorted.iloc[0]["Predicted Lead Time"], 2))

    st.subheader("Factory Comparison (Lead Time)")
    st.bar_chart(sim.sort_values("Predicted Lead Time").set_index("Factory")["Predicted Lead Time"])

    st.subheader("What-If Scenario Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current", round(current, 2))
    col2.metric("Optimized", round(best, 2))

    if lead_reduction > 0:
        col3.metric("Improvement %", f"{round(lead_reduction,2)}%", delta="Improved", delta_color="normal")
    else:
        col3.metric("Improvement %", f"{round(lead_reduction,2)}%", delta="Worse", delta_color="inverse")

    st.subheader("Recommendation Dashboard")

    top3 = sim_sorted.head(3).copy()
    top3["Improvement %"] = ((current - top3["Predicted Lead Time"]) / current) * 100

    st.dataframe(top3, use_container_width=True)

    st.subheader("Risk & Impact Panel")

    if profit_stability > 0.005:
        st.error("High profit variability risk")
    elif profit_stability > 0.002:
        st.warning("Moderate risk")
    else:
        st.success("Stable profit impact")

    if lead_reduction < 0:
        st.warning("No better factory available (current assignment is optimal)")
    elif lead_reduction > 5:
        st.success("High operational improvement")
    elif lead_reduction > 2:
        st.warning("Moderate improvement")
    else:
        st.error("Low improvement")


st.set_page_config(
    page_title="Unified Intelligence System",
    layout="wide"
)

apply_unified_theme()

selected = option_menu(
    menu_title=None,
    options=[
        "Profit Intelligence",
        "Shipping Intelligence",
        "Factory Optimization"
    ],
    icons=["bar-chart", "truck", "gear"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0.25rem 0.4rem",
            "background-color": "#eef3f7",
            "border": "1px solid #d3dde8",
            "border-radius": "12px"
        },
        "icon": {"color": "#134e4a", "font-size": "16px"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 12px",
            "color": "#1f2937",
            "--hover-color": "#dde6ee"
        },
        "nav-link-selected": {"background-color": "#dceeea", "color": "#134e4a"}
    }
)

st.sidebar.empty()

if selected == "Profit Intelligence":
    render_profit_intelligence()
elif selected == "Shipping Intelligence":
    render_shipping_intelligence()
elif selected == "Factory Optimization":
    render_factory_optimization()
