import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


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
        "Profit Intelligence Decision Support",
        "Executive view of product, division, and margin performance"
    )

    @st.cache_data
    def load_data():
        file_path = BASE_DIR / "Nassau Candy Distributor.xlsx"
        df = pd.read_excel(file_path)
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
            filtered_df["Product Name"].str.contains(
                product_search,
                case=False,
                na=False,
            )
        ]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Adjust the filters to continue.")
        return

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

    top_division = division_summary.sort_values(
        "Gross Profit", ascending=False
    ).iloc[0]
    weakest_division = division_summary.sort_values(
        "Gross Margin %"
    ).iloc[0]
    strongest_division = division_summary.sort_values(
        "Gross Margin %", ascending=False
    ).iloc[0]
    risk_products = product_summary[
        product_summary["Gross Margin %"] < margin_threshold
    ]

    executive_insights = [
        (
            "Profit Concentration",
            f"{top_division['Division']} contributes {safe_divide(top_division['Gross Profit'], total_profit) * 100:.1f}% of total profit.",
        ),
        (
            "Profit Drivers",
            f"{len(products_80)} products generate 80% of total profit in the current filtered view.",
        ),
    ]

    if not risk_products.empty:
        lowest_margin_product = risk_products.sort_values(
            "Gross Margin %"
        ).iloc[0]
        executive_insights.append(
            (
                "Pricing Review",
                f"{lowest_margin_product['Product Name']} is below the selected margin threshold and may require pricing or cost review.",
            )
        )
    else:
        executive_insights.append(
            (
                "Margin Health",
                "No products are currently below the selected margin threshold.",
            )
        )

    if len(division_summary) > 1:
        executive_insights.append(
            (
                "Division Efficiency",
                f"{weakest_division['Division']} shows lower margin efficiency than {strongest_division['Division']}.",
            )
        )
    else:
        executive_insights.append(
            (
                "Division Focus",
                f"{strongest_division['Division']} is the active division in the current view.",
            )
        )

    st.subheader("Executive Summary")
    summary_cols = st.columns(4)
    for column, (title, message) in zip(summary_cols, executive_insights):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(message)

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

    st.metric("Products Below Margin Threshold",
              len(risk_products))

    st.dataframe(risk_products)

    st.divider()

    st.header("Profit Concentration Analysis")

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
        df = pd.read_excel(BASE_DIR / "cleaned_nassau_shipping_data.xlsx")

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

    if filtered_df.empty:
        render_page_header(
            "Shipping Intelligence Decision Support",
            "Executive view of delivery speed, route performance, and geography"
        )
        st.warning("No shipments match the selected filters. Adjust the filters to continue.")
        return

    render_page_header(
        "Shipping Intelligence Decision Support",
        "Executive view of delivery speed, route performance, and geography"
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

    lead_time_span = route_perf["avg_lead_time"].max() - route_perf["avg_lead_time"].min()
    if route_perf.empty or lead_time_span == 0:
        route_perf["Efficiency Score"] = 1.0
    else:
        route_perf["Efficiency Score"] = 1 - (
            (route_perf["avg_lead_time"] - route_perf["avg_lead_time"].min()) /
            lead_time_span
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

    geographic_state_perf = filtered_df.groupby("State/Province").agg(
        avg_lead_time=("Shipping Lead Time", "mean"),
        shipments=("Order ID", "count")
    ).reset_index()
    region_perf = filtered_df.groupby("Region").agg(
        avg_lead_time=("Shipping Lead Time", "mean"),
        delay_rate=("Shipping Lead Time", lambda values: float((values > delay_threshold).mean() * 100)),
        shipments=("Order ID", "count")
    ).reset_index()

    if not geographic_state_perf.empty:
        best_state = geographic_state_perf.sort_values("avg_lead_time").iloc[0]
        worst_state = geographic_state_perf.sort_values("avg_lead_time", ascending=False).iloc[0]
    else:
        best_state = pd.Series({"State/Province": "N/A", "avg_lead_time": 0.0})
        worst_state = pd.Series({"State/Province": "N/A", "avg_lead_time": 0.0})

    if not region_perf.empty:
        most_efficient_region = region_perf.sort_values("avg_lead_time").iloc[0]
        highest_delay_region = region_perf.sort_values("delay_rate", ascending=False).iloc[0]
    else:
        most_efficient_region = pd.Series({"Region": "N/A", "avg_lead_time": 0.0})
        highest_delay_region = pd.Series({"Region": "N/A", "delay_rate": 0.0})

    st.subheader("Geographic Insights")
    geo1, geo2, geo3, geo4, geo5 = st.columns(5)
    geo1.metric("Best Performing State", best_state["State/Province"], delta=f"{best_state['avg_lead_time']:.1f} days")
    geo2.metric("Worst Performing State", worst_state["State/Province"], delta=f"{worst_state['avg_lead_time']:.1f} days")
    geo3.metric("Most Efficient Region", most_efficient_region["Region"], delta=f"{most_efficient_region['avg_lead_time']:.1f} days")
    geo4.metric("Highest Delay Region", highest_delay_region["Region"], delta=f"{highest_delay_region['delay_rate']:.1f}% delayed")
    geo5.metric("Average Lead Time", f"{avg_lead_time:.1f} days")

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


STATE_PROVINCE_COORDS = {
    "Alabama": (32.806671, -86.79113),
    "Alberta": (53.9333, -116.5765),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "British Columbia": (53.7267, -127.6476),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.9072, -77.0369),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.5266, -96.726486),
    "Kentucky": (37.66814, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Manitoba": (53.7609, -98.8139),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.12537, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Brunswick": (46.5653, -66.4619),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "Newfoundland and Labrador": (53.1355, -57.6604),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Nova Scotia": (44.682, -63.7443),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Ontario": (50.0, -85.0),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Prince Edward Island": (46.5107, -63.4168),
    "Quebec": (52.9399, -73.5491),
    "Rhode Island": (41.680893, -71.51178),
    "Saskatchewan": (52.9399, -106.4509),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954456),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.30249),
}


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(numerator / denominator)


def haversine_km(origin_lat, origin_lon, dest_lat, dest_lon):
    origin_lat = np.asarray(origin_lat, dtype=float)
    origin_lon = np.asarray(origin_lon, dtype=float)
    dest_lat = np.asarray(dest_lat, dtype=float)
    dest_lon = np.asarray(dest_lon, dtype=float)

    lat1 = np.radians(origin_lat)
    lon1 = np.radians(origin_lon)
    lat2 = np.radians(dest_lat)
    lon2 = np.radians(dest_lon)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    hav_component = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * (2 * np.arcsin(np.sqrt(hav_component)))


def normalize_benefit(series):
    cleaned = series.fillna(0).clip(lower=0)
    max_value = cleaned.max()
    if max_value <= 0:
        return pd.Series(0.0, index=cleaned.index)
    return cleaned / max_value


def percentile_rank(value, population):
    valid = pd.Series(population).dropna()
    if valid.empty:
        return 0.0
    return float((valid <= value).mean())


@st.cache_data(show_spinner=False)
def load_factory_optimization_data():
    df = pd.read_excel(BASE_DIR / "cleaned_nassau_shipping_data.xlsx")

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Lead Time"] = pd.to_numeric(df["Shipping Lead Time"], errors="coerce")
    df["Profit Margin"] = df["Gross Profit"] / df["Sales"].replace(0, np.nan)
    df["Profit per Unit"] = df["Gross Profit"] / df["Units"].replace(0, np.nan)

    geo_lookup = pd.DataFrame.from_dict(
        STATE_PROVINCE_COORDS,
        orient="index",
        columns=["customer_lat", "customer_lon"],
    )

    df = df.merge(
        geo_lookup,
        left_on="State/Province",
        right_index=True,
        how="left",
    )

    geo_ready = df.dropna(subset=["customer_lat", "customer_lon"]).copy()
    geo_ready["coord_weight"] = geo_ready["Units"].fillna(0).clip(lower=1)

    factory_coords = (
        geo_ready.groupby("Factory")[["customer_lat", "customer_lon", "coord_weight"]]
        .apply(
            lambda group: pd.Series(
                {
                    "factory_lat": np.average(
                        group["customer_lat"],
                        weights=group["coord_weight"],
                    ),
                    "factory_lon": np.average(
                        group["customer_lon"],
                        weights=group["coord_weight"],
                    ),
                }
            )
        )
        .reset_index()
    )

    df = df.merge(factory_coords, on="Factory", how="left")
    df["Route Distance KM"] = haversine_km(
        df["factory_lat"],
        df["factory_lon"],
        df["customer_lat"],
        df["customer_lon"],
    )

    product_factory_map = (
        df.groupby("Product Name")["Factory"]
        .agg(lambda values: values.mode().iat[0])
        .to_dict()
    )

    route_stats = (
        df.groupby(["Factory", "Region", "Ship Mode"])
        .agg(
            route_orders=("Order ID", "count"),
            lead_time_std=("Lead Time", lambda values: float(values.std(ddof=0))),
            margin_std=("Profit Margin", lambda values: float(values.std(ddof=0))),
            mean_distance_km=("Route Distance KM", "mean"),
        )
        .reset_index()
        .fillna({"lead_time_std": 0.0, "margin_std": 0.0, "mean_distance_km": 0.0})
    )

    factory_coord_map = {
        row["Factory"]: (row["factory_lat"], row["factory_lon"])
        for _, row in factory_coords.iterrows()
    }

    return df, product_factory_map, route_stats, factory_coord_map


@st.cache_resource(show_spinner=False)
def load_factory_model_artifacts():
    def load_pickle(file_name):
        with open(BASE_DIR / file_name, "rb") as artifact_file:
            return pickle.load(artifact_file)

    return {
        "model": load_pickle("leadtime_model.pkl"),
        "le_region": load_pickle("le_region.pkl"),
        "le_ship": load_pickle("le_ship.pkl"),
        "le_product": load_pickle("le_product.pkl"),
        "le_factory": load_pickle("le_factory.pkl"),
        "scaler": load_pickle("scaler.pkl"),
    }


@st.cache_data(show_spinner=False)
def benchmark_predictive_models():
    df, _, _, _ = load_factory_optimization_data()
    artifacts = load_factory_model_artifacts()

    feature_matrix = np.column_stack(
        [
            artifacts["le_region"].transform(df["Region"]),
            artifacts["le_ship"].transform(df["Ship Mode"]),
            artifacts["le_product"].transform(df["Product Name"]),
            artifacts["le_factory"].transform(df["Factory"]),
            artifacts["scaler"].transform(df[["Sales", "Units", "Cost"]]),
        ]
    )

    target = df["Lead Time"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        target,
        test_size=0.2,
        random_state=42,
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        ),
    }

    metrics = []
    for model_name, candidate_model in candidate_models.items():
        candidate_model.fit(X_train, y_train)
        predictions = candidate_model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        metrics.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["R²", "RMSE", "MAE"],
        ascending=[False, True, True],
    )

    best_row = metrics_df.iloc[0]
    target_std = float(np.std(y_test))
    target_mad = float(np.mean(np.abs(y_test - np.median(y_test))))
    rmse_quality = 1 / (1 + safe_divide(best_row["RMSE"], target_std))
    mae_quality = 1 / (1 + safe_divide(best_row["MAE"], target_mad))
    r2_quality = float(np.clip(best_row["R²"], 0, 1))
    confidence_score = 100 * np.mean([rmse_quality, mae_quality, r2_quality])

    return metrics_df.reset_index(drop=True), best_row["Model"], confidence_score


@st.cache_resource(show_spinner=False)
def build_profit_sensitivity_model():
    df, _, _, _ = load_factory_optimization_data()
    profit_model = GradientBoostingRegressor(random_state=42)
    profit_model.fit(
        df[["Sales", "Cost", "Units", "Lead Time"]],
        df["Profit Margin"],
    )
    return profit_model


def select_scenario_subset(df, product, region, shipmode):
    candidate_filters = [
        (
            "Product + Region + Ship Mode",
            (df["Product Name"] == product)
            & (df["Region"] == region)
            & (df["Ship Mode"] == shipmode),
        ),
        (
            "Product + Region",
            (df["Product Name"] == product) & (df["Region"] == region),
        ),
        (
            "Product + Ship Mode",
            (df["Product Name"] == product) & (df["Ship Mode"] == shipmode),
        ),
        ("Product", df["Product Name"] == product),
    ]

    for scope_name, mask in candidate_filters:
        subset = df.loc[mask].copy()
        if not subset.empty:
            return subset, scope_name

    return df.loc[df["Product Name"] == product].copy(), "Product"


def get_factory_history(df, factory_name, region, shipmode):
    candidate_filters = [
        (
            "Factory + Region + Ship Mode",
            (df["Factory"] == factory_name)
            & (df["Region"] == region)
            & (df["Ship Mode"] == shipmode),
        ),
        (
            "Factory + Region",
            (df["Factory"] == factory_name) & (df["Region"] == region),
        ),
        ("Factory", df["Factory"] == factory_name),
    ]

    for scope_name, mask in candidate_filters:
        subset = df.loc[mask].copy()
        if not subset.empty:
            return subset, scope_name

    return df.loc[df["Factory"] == factory_name].copy(), "Factory"


def predict_lead_time(artifacts, product, region, shipmode, factory, sales, units, cost):
    scaled_numeric = artifacts["scaler"].transform(
        pd.DataFrame(
            [[sales, units, cost]],
            columns=["Sales", "Units", "Cost"],
        )
    )
    encoded_row = pd.DataFrame(
        [
            {
                "Region_enc": artifacts["le_region"].transform([region])[0],
                "ShipMode_enc": artifacts["le_ship"].transform([shipmode])[0],
                "Product_enc": artifacts["le_product"].transform([product])[0],
                "Factory_enc": artifacts["le_factory"].transform([factory])[0],
                "Sales": scaled_numeric[0][0],
                "Units": scaled_numeric[0][1],
                "Cost": scaled_numeric[0][2],
            }
        ]
    )
    return float(artifacts["model"].predict(encoded_row)[0])


def estimate_profit_impact(
    profit_model,
    sales_mean,
    cost_mean,
    units_mean,
    current_lead_time,
    proposed_lead_time,
    total_sales,
):
    feature_columns = ["Sales", "Cost", "Units", "Lead Time"]
    current_features = pd.DataFrame(
        [[sales_mean, cost_mean, units_mean, current_lead_time]],
        columns=feature_columns,
    )
    proposed_features = pd.DataFrame(
        [[sales_mean, cost_mean, units_mean, proposed_lead_time]],
        columns=feature_columns,
    )

    current_margin = float(profit_model.predict(current_features)[0])
    proposed_margin = float(profit_model.predict(proposed_features)[0])

    current_profit = current_margin * total_sales
    proposed_profit = proposed_margin * total_sales
    impact_value = proposed_profit - current_profit
    impact_pct = safe_divide(impact_value, current_profit) * 100

    return current_profit, proposed_profit, impact_value, impact_pct


def average_route_distance(factory_name, scenario_df, factory_coord_map):
    if factory_name not in factory_coord_map:
        return np.nan

    geo_points = scenario_df.dropna(subset=["customer_lat", "customer_lon"])
    if geo_points.empty:
        return np.nan

    factory_lat, factory_lon = factory_coord_map[factory_name]
    distances = haversine_km(
        factory_lat,
        factory_lon,
        geo_points["customer_lat"].to_numpy(),
        geo_points["customer_lon"].to_numpy(),
    )
    return float(np.nanmean(distances))


def compute_route_risk(route_history, route_stats, baseline_margin):
    order_count = int(route_history["Order ID"].count())
    lead_time_std = float(route_history["Lead Time"].std(ddof=0) or 0.0)
    margin_std = float(route_history["Profit Margin"].std(ddof=0) or 0.0)
    downside_rate = float((route_history["Profit Margin"] < baseline_margin).mean())
    lead_rank = percentile_rank(lead_time_std, route_stats["lead_time_std"])
    margin_rank = percentile_rank(margin_std, route_stats["margin_std"])
    order_rank = percentile_rank(order_count, route_stats["route_orders"])

    risk_score = 100 * np.mean([lead_rank, margin_rank, downside_rate, 1 - order_rank])
    stability_score = 100 * np.mean(
        [1 - lead_rank, 1 - margin_rank, order_rank, 1 - downside_rate]
    )

    return float(risk_score), float(stability_score), order_count


def assign_cluster_labels(centroids):
    centroids = centroids.copy()
    centroids["efficiency_index"] = centroids["gross_profit"] / centroids["lead_time"].clip(lower=1)
    centroids["congestion_index"] = centroids["lead_time"] * centroids["units"]

    labels = {cluster_id: "Moderate Performance Routes" for cluster_id in centroids["cluster"]}
    used_clusters = set()

    ranking_rules = [
        (
            "Best Performing Routes",
            centroids.sort_values("efficiency_index", ascending=False)["cluster"],
        ),
        (
            "High Congestion Routes",
            centroids.sort_values("congestion_index", ascending=False)["cluster"],
        ),
    ]

    for label, ranked_clusters in ranking_rules:
        for cluster_id in ranked_clusters:
            if cluster_id not in used_clusters:
                labels[cluster_id] = label
                used_clusters.add(cluster_id)
                break

    return labels


def build_cluster_frames(df):
    route_cluster_df = (
        df.groupby(["Factory", "State/Province", "Region", "Route"])
        .agg(
            lead_time=("Lead Time", "mean"),
            sales=("Sales", "sum"),
            gross_profit=("Gross Profit", "sum"),
            units=("Units", "sum"),
            orders=("Order ID", "count"),
            distance_km=("Route Distance KM", "mean"),
        )
        .reset_index()
    )
    route_cluster_df["region_code"] = route_cluster_df["Region"].astype("category").cat.codes

    if len(route_cluster_df) >= 2:
        route_features = ["lead_time", "sales", "gross_profit", "units", "region_code"]
        route_scaler = StandardScaler()
        route_matrix = route_scaler.fit_transform(route_cluster_df[route_features])
        route_cluster_count = min(4, len(route_cluster_df))
        route_model = KMeans(n_clusters=route_cluster_count, random_state=42, n_init=10)
        route_cluster_df["cluster"] = route_model.fit_predict(route_matrix)

        route_centroids = pd.DataFrame(
            route_scaler.inverse_transform(route_model.cluster_centers_),
            columns=route_features,
        )
        route_centroids["cluster"] = range(route_cluster_count)
        route_labels = assign_cluster_labels(route_centroids)
        route_cluster_df["Cluster Label"] = route_cluster_df["cluster"].map(route_labels)
    else:
        route_cluster_df["cluster"] = 0
        route_cluster_df["Cluster Label"] = "Moderate Performance Routes"

    product_cluster_df = (
        df.groupby(["Product Name", "Region"])
        .agg(
            lead_time=("Lead Time", "mean"),
            sales=("Sales", "sum"),
            gross_profit=("Gross Profit", "sum"),
            units=("Units", "sum"),
            orders=("Order ID", "count"),
        )
        .reset_index()
    )
    product_cluster_df["region_code"] = product_cluster_df["Region"].astype("category").cat.codes

    if len(product_cluster_df) >= 2:
        product_features = ["lead_time", "sales", "gross_profit", "units", "region_code"]
        product_scaler = StandardScaler()
        product_matrix = product_scaler.fit_transform(product_cluster_df[product_features])
        product_cluster_count = min(4, len(product_cluster_df))
        product_model = KMeans(
            n_clusters=product_cluster_count,
            random_state=42,
            n_init=10,
        )
        product_cluster_df["cluster"] = product_model.fit_predict(product_matrix)

        product_centroids = pd.DataFrame(
            product_scaler.inverse_transform(product_model.cluster_centers_),
            columns=product_features,
        )
        product_centroids["cluster"] = range(product_cluster_count)
        product_labels = assign_cluster_labels(product_centroids)
        product_cluster_df["Cluster Label"] = product_cluster_df["cluster"].map(product_labels)
    else:
        product_cluster_df["cluster"] = 0
        product_cluster_df["Cluster Label"] = "Moderate Performance Routes"

    return route_cluster_df, product_cluster_df


def classify_operational_risk(risk_score):
    if risk_score >= 67:
        return "High"
    if risk_score >= 34:
        return "Moderate"
    return "Low"


def classify_financial_stability(stability_score):
    if stability_score >= 70:
        return "High"
    if stability_score >= 45:
        return "Medium"
    return "Low"


def classify_recommendation_confidence(confidence_score):
    if confidence_score >= 65:
        return "High Confidence"
    if confidence_score >= 35:
        return "Moderate Confidence"
    return "Low Confidence"


def format_business_impact(value):
    if value > 0:
        return f"+{value:.2f}%"
    if value < 0:
        return f"{value:.2f}%"
    return "0.00%"


def format_variance_text(value, improvement_word, decline_word):
    if value > 0:
        return f"{value:.1f}% {improvement_word}"
    if value < 0:
        return f"+{abs(value):.1f}% {decline_word}"
    return "No change"


def build_rejection_reason(option_row):
    reasons = []
    if option_row["Lead Time Reduction %"] < 0:
        reasons.append("slower lead time")
    if option_row["Distance Reduction %"] < 0:
        reasons.append("longer shipping distance")
    if option_row["Profit Impact %"] < 0:
        reasons.append("lower profitability")
    if not reasons:
        reasons.append("no clear business advantage")
    return "Rejected - " + ", ".join(reasons)


def build_recommendation_explanation(
    recommended_factory,
    current_factory,
    recommended_row,
    no_better_option,
):
    if no_better_option:
        return (
            f"{current_factory} remains the best option because the alternative factories "
            "do not improve lead time, shipping distance, and profitability at the same time."
        )

    explanation_parts = []
    if recommended_row["Lead Time Reduction %"] > 0:
        explanation_parts.append(
            f"reduces lead time by {recommended_row['Lead Time Reduction %']:.1f}%"
        )
    if recommended_row["Distance Reduction %"] > 0:
        explanation_parts.append(
            f"cuts shipping distance by {recommended_row['Distance Reduction %']:.1f}%"
        )
    if recommended_row["Profit Impact %"] > 0:
        explanation_parts.append(
            f"improves expected profit by {recommended_row['Profit Impact %']:.2f}%"
        )

    if not explanation_parts:
        explanation_parts.append("maintains the current performance level")

    if len(explanation_parts) == 1:
        joined_explanation = explanation_parts[0]
    else:
        joined_explanation = ", ".join(explanation_parts[:-1]) + f", and {explanation_parts[-1]}"

    operational_risk = classify_operational_risk(recommended_row["Risk Score"]).lower()
    return (
        f"{recommended_factory} is recommended because it {joined_explanation} "
        f"while keeping operational risk {operational_risk}."
    )


def render_factory_optimization():
    df, product_factory_map, route_stats, factory_coord_map = load_factory_optimization_data()
    artifacts = load_factory_model_artifacts()
    profit_model = build_profit_sensitivity_model()
    benchmark_metrics, best_model_name, confidence_score = benchmark_predictive_models()
    route_cluster_df, product_cluster_df = build_cluster_frames(df)

    factories = list(artifacts["le_factory"].classes_)

    st.sidebar.header("Filters & Controls")

    product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
    region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
    shipmode = st.sidebar.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))

    priority = st.sidebar.slider(
        "Optimization Priority (Speed ←→ Profit)",
        0, 100, 50
    )

    current_factory = product_factory_map[product]
    scenario_df, scenario_scope = select_scenario_subset(df, product, region, shipmode)
    current_route_df = scenario_df[scenario_df["Factory"] == current_factory].copy()
    if current_route_df.empty:
        current_route_df, _ = get_factory_history(df, current_factory, region, shipmode)

    sales_mean = float(scenario_df["Sales"].mean())
    units_mean = float(scenario_df["Units"].mean())
    cost_mean = float(scenario_df["Cost"].mean())
    total_sales = float(scenario_df["Sales"].sum())
    baseline_margin = float(current_route_df["Profit Margin"].mean())
    current_lead_time = float(current_route_df["Lead Time"].mean())
    current_distance = average_route_distance(current_factory, scenario_df, factory_coord_map)
    current_efficiency = safe_divide(1.0, current_distance * current_lead_time)
    baseline_profit_before, baseline_profit_after, _, _ = estimate_profit_impact(
        profit_model,
        sales_mean,
        cost_mean,
        units_mean,
        current_lead_time,
        current_lead_time,
        total_sales,
    )

    current_history, current_support_scope = get_factory_history(
        df,
        current_factory,
        region,
        shipmode,
    )
    current_risk, current_stability, current_support = compute_route_risk(
        current_history,
        route_stats,
        baseline_margin,
    )

    simulation_rows = []
    for factory_name in factories:
        if factory_name == current_factory:
            predicted_lead_time = current_lead_time
            proposed_distance = current_distance
            expected_profit_before = baseline_profit_before
            expected_profit_after = baseline_profit_after
            profit_impact = 0.0
            profit_impact_pct = 0.0
            efficiency_improvement_pct = 0.0
            history_scope = current_support_scope
            risk_score = current_risk
            stability_score = current_stability
            support_count = current_support
        else:
            predicted_lead_time = predict_lead_time(
                artifacts,
                product,
                region,
                shipmode,
                factory_name,
                sales_mean,
                units_mean,
                cost_mean,
            )
            proposed_distance = average_route_distance(factory_name, scenario_df, factory_coord_map)
            expected_profit_before, expected_profit_after, profit_impact, profit_impact_pct = estimate_profit_impact(
                profit_model,
                sales_mean,
                cost_mean,
                units_mean,
                current_lead_time,
                predicted_lead_time,
                total_sales,
            )
            proposed_efficiency = safe_divide(1.0, proposed_distance * predicted_lead_time)
            efficiency_improvement_pct = safe_divide(
                proposed_efficiency - current_efficiency,
                current_efficiency,
            ) * 100
            history_slice, history_scope = get_factory_history(df, factory_name, region, shipmode)
            risk_score, stability_score, support_count = compute_route_risk(
                history_slice,
                route_stats,
                baseline_margin,
            )

        lead_time_difference = predicted_lead_time - current_lead_time
        distance_difference = proposed_distance - current_distance
        lead_time_reduction_pct = safe_divide(
            current_lead_time - predicted_lead_time,
            current_lead_time,
        ) * 100
        distance_reduction_pct = safe_divide(
            current_distance - proposed_distance,
            current_distance,
        ) * 100

        simulation_rows.append(
            {
                "Factory": factory_name,
                "Factory Role": "Current" if factory_name == current_factory else "Alternative",
                "Support Scope": history_scope,
                "Support Orders": support_count,
                "Predicted Lead Time": predicted_lead_time,
                "Lead Time Difference": lead_time_difference,
                "Lead Time Reduction %": lead_time_reduction_pct,
                "Current Distance KM": current_distance,
                "Recommended Distance KM": proposed_distance,
                "Distance Difference KM": distance_difference,
                "Distance Reduction %": distance_reduction_pct,
                "Shipping Efficiency Improvement %": efficiency_improvement_pct,
                "Expected Profit Before": expected_profit_before,
                "Expected Profit After": expected_profit_after,
                "Profit Impact": profit_impact,
                "Profit Impact %": profit_impact_pct,
                "Risk Score": risk_score,
                "Financial Stability Score": stability_score,
            }
        )

    sim = pd.DataFrame(simulation_rows)
    sim["Risk Improvement %"] = sim["Risk Score"].apply(
        lambda value: safe_divide(current_risk - value, current_risk) * 100
    )

    speed_weight = 1 - (priority / 100)
    profit_weight = priority / 100

    sim["Lead Benefit"] = normalize_benefit(sim["Lead Time Reduction %"])
    sim["Distance Benefit"] = normalize_benefit(sim["Distance Reduction %"])
    sim["Profit Benefit"] = normalize_benefit(sim["Profit Impact %"])
    sim["Risk Benefit"] = normalize_benefit(sim["Risk Improvement %"])

    sim["Optimization Score"] = 100 * (
        speed_weight * (0.5 * sim["Lead Benefit"] + 0.5 * sim["Distance Benefit"])
        + profit_weight * (0.5 * sim["Profit Benefit"] + 0.5 * sim["Risk Benefit"])
    )

    sim["Recommendation Eligible"] = (
        (sim["Factory"] != current_factory)
        & (sim["Lead Time Difference"] <= 0)
        & (sim["Distance Difference KM"] <= 0)
        & (sim["Profit Impact"] >= 0)
        & (
            (sim["Lead Time Reduction %"] > 0)
            | (sim["Distance Reduction %"] > 0)
            | (sim["Profit Impact %"] > 0)
        )
    )

    sim_sorted = sim.sort_values("Optimization Score", ascending=False).reset_index(drop=True)
    candidate_recommendations = sim_sorted[sim_sorted["Recommendation Eligible"]].copy()
    recommendation_coverage = safe_divide(
        len(candidate_recommendations),
        max(len(factories) - 1, 1),
    ) * 100

    if candidate_recommendations.empty:
        recommended_factory = current_factory
        recommended_row = sim.loc[sim["Factory"] == current_factory].iloc[0]
        no_better_option = True
    else:
        recommended_row = candidate_recommendations.iloc[0]
        recommended_factory = recommended_row["Factory"]
        no_better_option = False

    recommendation_confidence = confidence_score
    recommended_risk_level = classify_operational_risk(recommended_row["Risk Score"])
    financial_stability_label = classify_financial_stability(
        recommended_row["Financial Stability Score"]
    )
    recommendation_confidence_label = classify_recommendation_confidence(
        recommendation_confidence
    )
    recommendation_explanation = build_recommendation_explanation(
        recommended_factory,
        current_factory,
        recommended_row,
        no_better_option,
    )

    comparison_table = sim_sorted.copy()
    comparison_table["Risk Level"] = comparison_table["Risk Score"].apply(
        lambda value: f"{classify_operational_risk(value)} Operational Risk"
    )
    comparison_table["Factory Display"] = comparison_table["Factory"].apply(
        lambda name: f"{name} (Recommended)" if name == recommended_factory else name
    )
    comparison_table["Sort Priority"] = (
        comparison_table["Factory"] != recommended_factory
    ).astype(int)
    comparison_table = comparison_table.sort_values(
        ["Sort Priority", "Optimization Score"],
        ascending=[True, False],
    )

    if no_better_option:
        rejected_options = comparison_table[
            comparison_table["Factory"] != current_factory
        ].copy()
    else:
        rejected_options = comparison_table[
            (~comparison_table["Factory"].isin([current_factory, recommended_factory]))
        ].copy()

    rejected_options["Lead Time Difference Display"] = rejected_options[
        "Lead Time Reduction %"
    ].apply(lambda value: format_variance_text(value, "faster", "slower"))
    rejected_options["Distance Difference Display"] = rejected_options[
        "Distance Reduction %"
    ].apply(lambda value: format_variance_text(value, "shorter", "longer distance"))
    rejected_options["Reason"] = rejected_options.apply(build_rejection_reason, axis=1)

    best_metric_row = benchmark_metrics.loc[
        benchmark_metrics["Model"] == best_model_name
    ].iloc[0]

    if no_better_option:
        executive_insight_message = (
            "No factory reassignment is recommended because the current factory remains the most efficient option across lead time, distance, profitability, and operational stability."
        )
    else:
        executive_insight_message = (
            "Factory reassignment is recommended. "
            f"Estimated lead time reduction: {recommended_row['Lead Time Reduction %']:.1f}%. "
            f"Estimated distance reduction: {recommended_row['Distance Reduction %']:.1f}%. "
            f"Estimated profit impact: {format_business_impact(recommended_row['Profit Impact %'])}."
        )

    render_page_header(
        "Factory Optimization Decision Support",
        "Business view of factory assignment recommendations"
    )

    st.subheader("Executive KPI Summary")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric(
            "Lead Time Reduction",
            f"{recommended_row['Lead Time Reduction %']:.2f}%",
            help="Expected reduction in delivery lead time compared with the current factory.",
        )
    with k2:
        st.metric(
            "Distance Reduction",
            f"{recommended_row['Distance Reduction %']:.2f}%",
            help="Expected reduction in shipping distance compared with the current factory.",
        )
    with k3:
        st.metric(
            "Profit Impact",
            format_business_impact(recommended_row["Profit Impact %"]),
            help="Estimated change in profitability if the recommendation is applied.",
        )
    with k4:
        st.metric(
            "Financial Stability",
            financial_stability_label,
            help="Shows how stable the expected financial outcome is for this factory option.",
        )
        st.caption(f"Score: {recommended_row['Financial Stability Score']:.1f}")
    with k5:
        st.metric(
            "Recommendation Confidence",
            recommendation_confidence_label,
            help="Shows how dependable the recommendation is based on historical validation performance.",
        )
        st.caption(f"Score: {recommendation_confidence:.1f}%")
    with k6:
        st.metric(
            "Recommendation Coverage",
            f"{recommendation_coverage:.2f}%",
            help="Share of evaluated alternative factories that met the business recommendation rules.",
        )

    st.markdown("---")

    if no_better_option:
        st.success(
            "Current Assignment Already Optimal\n\n"
            "All available factories were evaluated. The current factory delivers the best combination of shipping speed, transportation distance, profitability, and operational stability. No reassignment is recommended."
        )
    else:
        st.info(executive_insight_message)

    st.markdown("---")

    st.subheader("Recommendation Summary")
    with st.container(border=True):
        if no_better_option:
            st.markdown("#### No Better Factory Available")
            st.write(
                "All alternative factories were evaluated. The current factory remains the most efficient option based on lead time, distance, and profitability."
            )
        else:
            st.markdown("#### Recommended Factory Reassignment")
            st.write("This recommendation gives the clearest overall business improvement for the selected product and route.")

        summary_top_left, summary_top_right = st.columns(2)
        summary_top_left.metric("Current Factory", current_factory)
        summary_top_right.metric("Recommended Factory", recommended_factory)

        comparison_metrics = st.columns(4)
        comparison_metrics[0].metric("Current Lead Time", f"{current_lead_time:.2f}")
        comparison_metrics[1].metric("Recommended Lead Time", f"{recommended_row['Predicted Lead Time']:.2f}")
        comparison_metrics[2].metric("Current Distance", f"{current_distance:.2f} km")
        comparison_metrics[3].metric("Recommended Distance", f"{recommended_row['Recommended Distance KM']:.2f} km")

        improvement_metrics = st.columns(3)
        improvement_metrics[0].metric(
            "Expected Lead Time Improvement",
            f"{recommended_row['Lead Time Reduction %']:.2f}%",
        )
        improvement_metrics[1].metric(
            "Expected Distance Reduction",
            f"{recommended_row['Distance Reduction %']:.2f}%",
        )
        improvement_metrics[2].metric(
            "Expected Profit Impact",
            format_business_impact(recommended_row["Profit Impact %"]),
        )

        st.markdown("**Business Summary**")
        st.write(recommendation_explanation)

    st.markdown("---")

    st.subheader("Why Other Factories Were Not Selected")
    st.caption("Alternatives are ranked from the strongest remaining option downward.")
    st.dataframe(
        rejected_options[
            [
                "Factory",
                "Lead Time Difference Display",
                "Distance Difference Display",
                "Reason",
            ]
        ].rename(
            columns={
                "Lead Time Difference Display": "Lead Time Difference",
                "Distance Difference Display": "Distance Difference",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Factory Comparison")
    st.caption("Click any column header to sort the comparison.")
    st.dataframe(
        comparison_table[
            [
                "Factory Display",
                "Predicted Lead Time",
                "Recommended Distance KM",
                "Lead Time Reduction %",
                "Distance Reduction %",
                "Profit Impact %",
                "Risk Level",
            ]
        ].rename(
            columns={
                "Factory Display": "Factory",
                "Recommended Distance KM": "Distance",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Predicted Lead Time": st.column_config.NumberColumn(format="%.2f"),
            "Distance": st.column_config.NumberColumn(format="%.2f km"),
            "Lead Time Reduction %": st.column_config.NumberColumn(format="%.2f%%"),
            "Distance Reduction %": st.column_config.NumberColumn(format="%.2f%%"),
            "Profit Impact %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    st.markdown("---")

    st.subheader("Why This Recommendation?")
    with st.container(border=True):
        st.write(recommendation_explanation)
        if no_better_option:
            st.write(
                "The current assignment stays in place because the alternative factories do not deliver a better business outcome across speed, distance, and profitability."
            )
        else:
            st.write(
                f"{recommended_factory} leads the ranked options while keeping profitability stable and operational risk {recommended_risk_level.lower()}."
            )

    st.markdown("---")

    st.subheader("Risk & Alerts")
    alert_left, alert_middle, alert_right = st.columns(3)

    with alert_left:
        if recommended_risk_level == "High":
            st.error("High operational risk\n\nThis option shows more delivery variability than most alternatives and should be monitored closely.")
        elif recommended_risk_level == "Moderate":
            st.warning("Moderate operational risk\n\nThis option is workable but should remain under regular operational review.")
        else:
            st.success("Low operational risk\n\nThis option is comparatively stable based on historical delivery performance.")

    with alert_middle:
        if no_better_option:
            st.info("Current assignment already optimal\n\nNo alternative factory clears the business rules for a better recommendation.")
        elif (
            recommended_row["Lead Time Reduction %"] >= 10
            or recommended_row["Distance Reduction %"] >= 10
            or recommended_row["Profit Impact %"] >= 1
        ):
            st.success("Significant improvement opportunity identified\n\nThe recommended factory offers a meaningful operational improvement.")
        else:
            st.warning("Moderate improvement opportunity identified\n\nThe recommended factory improves performance, but the gain is incremental.")

    with alert_right:
        if recommended_row["Financial Stability Score"] >= 70:
            st.success("Financial outlook remains stable\n\nThe recommendation supports a consistent profitability profile.")
        elif recommended_row["Financial Stability Score"] >= 45:
            st.warning("Financial outlook should be monitored\n\nThe recommendation is operationally viable but deserves margin tracking.")
        else:
            st.error("Financial outlook is less stable\n\nApply extra review before making a factory reassignment decision.")

    with st.expander("Advanced Analytics & Technical Validation"):

        st.subheader("Before vs After Comparison")
        before_after_top = st.columns(4)
        before_after_top[0].metric("Current Lead Time", f"{current_lead_time:.2f}")
        before_after_top[1].metric("Recommended Lead Time", f"{recommended_row['Predicted Lead Time']:.2f}")
        before_after_top[2].metric("Current Distance", f"{current_distance:.2f} km")
        before_after_top[3].metric("Recommended Distance", f"{recommended_row['Recommended Distance KM']:.2f} km")

        before_after_bottom = st.columns(3)
        before_after_bottom[0].metric("Lead Time Improvement", f"{recommended_row['Lead Time Reduction %']:.2f}%")
        before_after_bottom[1].metric("Distance Improvement", f"{recommended_row['Distance Reduction %']:.2f}%")
        before_after_bottom[2].metric("Profit Impact", format_business_impact(recommended_row["Profit Impact %"]))

        st.subheader("Detailed Comparison Visuals")
        comparison_left, comparison_right = st.columns(2)

        with comparison_left:
            fig_compare = px.scatter(
                sim_sorted,
                x="Predicted Lead Time",
                y="Recommended Distance KM",
                size=sim_sorted["Profit Impact"].abs() + 1,
                color="Optimization Score",
                hover_name="Factory",
                hover_data={
                    "Lead Time Reduction %": ":.2f",
                    "Distance Reduction %": ":.2f",
                    "Profit Impact %": ":.4f",
                    "Risk Score": ":.2f",
                },
                title="Lead Time vs Distance by Factory",
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with comparison_right:
            comparison_melt = sim_sorted.melt(
                id_vars=["Factory"],
                value_vars=[
                    "Predicted Lead Time",
                    "Recommended Distance KM",
                    "Profit Impact %",
                ],
                var_name="Metric",
                value_name="Value",
            )
            fig_factory_compare = px.bar(
                comparison_melt,
                x="Factory",
                y="Value",
                color="Metric",
                barmode="group",
                title="Lead Time, Distance, and Profit Impact Comparison",
            )
            st.plotly_chart(fig_factory_compare, use_container_width=True)

        st.subheader("Cluster Analysis")
        cluster_left, cluster_right = st.columns(2)

        with cluster_left:
            fig_route_clusters = px.scatter(
                route_cluster_df,
                x="distance_km",
                y="lead_time",
                color="Cluster Label",
                size="units",
                hover_name="Route",
                hover_data=["Factory", "State/Province", "orders"],
                title="Route Clusters",
            )
            st.plotly_chart(fig_route_clusters, use_container_width=True)
            st.caption(
                "Routes in the High Congestion group consistently carry longer delivery times and may need logistics redesign or capacity balancing."
            )

        with cluster_right:
            fig_product_clusters = px.scatter(
                product_cluster_df,
                x="sales",
                y="gross_profit",
                color="Cluster Label",
                size="units",
                hover_name="Product Name",
                hover_data=["Region", "lead_time", "orders"],
                title="Product Clusters",
            )
            st.plotly_chart(fig_product_clusters, use_container_width=True)
            st.caption(
                "Products in the Best Performing group show stronger route economics, while the Moderate and High Congestion groups signal where operational review may help."
            )

        congested_hotspots = (
            route_cluster_df[route_cluster_df["Cluster Label"] == "High Congestion Routes"]
            .groupby("State/Province")
            .agg(
                total_orders=("orders", "sum"),
                avg_lead_time=("lead_time", "mean"),
            )
            .reset_index()
            .sort_values(["total_orders", "avg_lead_time"], ascending=[False, False])
            .head(10)
        )

        if not congested_hotspots.empty:
            fig_hotspots = px.bar(
                congested_hotspots,
                x="State/Province",
                y="total_orders",
                color="avg_lead_time",
                title="Routes Requiring Attention",
            )
            st.plotly_chart(fig_hotspots, use_container_width=True)
            st.caption(
                "States with heavier volume and longer delivery times are grouped here because they show the strongest congestion signals."
            )

        st.subheader("Model Performance Dashboard")
        performance_metrics = st.columns(4)
        performance_metrics[0].metric("Best Model", best_model_name)
        performance_metrics[1].metric("RMSE", f"{best_metric_row['RMSE']:.2f}")
        performance_metrics[2].metric("MAE", f"{best_metric_row['MAE']:.2f}")
        performance_metrics[3].metric("R²", f"{best_metric_row['R²']:.4f}")

        model_left, model_right = st.columns([1.3, 1])

        with model_left:
            st.dataframe(benchmark_metrics, hide_index=True, use_container_width=True)

        with model_right:
            st.metric("Current Factory Support", f"{current_support} orders")
            st.metric("Current Risk Score", f"{current_risk:.2f}")
            st.metric("Current Financial Stability", f"{current_stability:.2f}")

        metrics_melt = benchmark_metrics.melt(
            id_vars="Model",
            value_vars=["RMSE", "MAE", "R²"],
            var_name="Metric",
            value_name="Value",
        )
        fig_model_metrics = px.bar(
            metrics_melt,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Predictive Model Metrics",
        )
        st.plotly_chart(fig_model_metrics, use_container_width=True)

        st.subheader("Technical Validation Metrics")
        technical_validation = pd.DataFrame(
            [
                {"Metric": "Scenario Scope", "Value": scenario_scope},
                {"Metric": "Current Factory Mapping", "Value": current_factory},
                {"Metric": "Recommended Factory", "Value": recommended_factory},
                {"Metric": "Current Support Orders", "Value": current_support},
                {"Metric": "Recommendation Eligible", "Value": "Yes" if not no_better_option else "No"},
                {"Metric": "Optimization Score", "Value": round(recommended_row["Optimization Score"], 2)},
            ]
        )
        st.dataframe(technical_validation, hide_index=True, use_container_width=True)


st.set_page_config(
    page_title="Nassau Candy Distributor",
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
