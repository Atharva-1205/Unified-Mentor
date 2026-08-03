from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "APL_Logistics.csv"

REAL_DAYS_COL = "Days for shipping (real)"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"
LATE_RISK_COL = "Late_delivery_risk"
STATUS_COL = "Delivery Status"
MODE_COL = "Shipping Mode"
REGION_COL = "Order Region"
MARKET_COL = "Market"
SEGMENT_COL = "Customer Segment"
COUNTRY_COL = "Order Country"
LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COLUMN_CANDIDATES = ["order date (DateOrders)", "Order Date", "DateOrders"]


def apply_theme():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        .page-title {
            color: inherit !important;
            font-size: 2.35rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stMetric"] {
            background: #f5f8fb;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            color: #0f172a !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricDelta"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
            color: #475569 !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #0f172a !important;
            opacity: 1 !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 1.18rem !important;
            font-weight: 700 !important;
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] p,
        div[data-testid="stTabs"] button[data-baseweb="tab"] div,
        div[data-testid="stTabs"] button[data-baseweb="tab"] span,
        div[data-testid="stTabs"] button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.18rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
        }
        .insight-card {
            border: 1px solid #d9e2ec;
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            height: 100%;
            color: #1f2937;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .insight-card * {
            color: inherit !important;
        }
        .insight-title {
            color: #0f172a !important;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .insight-body {
            color: #475569 !important;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_logistics_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
    source_rows = len(df)
    df.columns = df.columns.str.strip()

    text_columns = [
        STATUS_COL,
        MODE_COL,
        REGION_COL,
        MARKET_COL,
        SEGMENT_COL,
        COUNTRY_COL,
        "Order State",
        "Customer Country",
        "Customer State",
        "Category Name",
        "Department Name",
        "Product Name",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA})
            )

    numeric_columns = [
        REAL_DAYS_COL,
        SCHEDULED_DAYS_COL,
        LATE_RISK_COL,
        LAT_COL,
        LON_COL,
        "Sales",
        "Order Profit Per Order",
        "Benefit per order",
        "Order Item Quantity",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    date_column = next((col for col in DATE_COLUMN_CANDIDATES if col in df.columns), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    valid_mask = (
        df[REAL_DAYS_COL].notna()
        & df[SCHEDULED_DAYS_COL].notna()
        & df[LATE_RISK_COL].isin([0, 1])
        & df[STATUS_COL].notna()
        & df[MODE_COL].notna()
        & df[REGION_COL].notna()
        & df[MARKET_COL].notna()
        & df[SEGMENT_COL].notna()
        & (df[REAL_DAYS_COL] >= 0)
        & (df[SCHEDULED_DAYS_COL] >= 0)
    )

    if date_column:
        valid_mask &= df[date_column].notna()

    cleaned_df = df.loc[valid_mask].copy()
    cleaned_df["Delay Gap"] = cleaned_df[REAL_DAYS_COL] - cleaned_df[SCHEDULED_DAYS_COL]
    cleaned_df["Delivery Timing"] = np.select(
        [
            cleaned_df["Delay Gap"] < 0,
            cleaned_df["Delay Gap"] == 0,
            cleaned_df["Delay Gap"] > 0,
        ],
        ["Early", "On-time", "Delayed"],
        default="On-time",
    )
    cleaned_df["On-Time Delivery"] = cleaned_df["Delay Gap"] <= 0
    cleaned_df["Delayed Shipment"] = cleaned_df["Delay Gap"] > 0
    cleaned_df["Late Delivery Risk"] = cleaned_df[LATE_RISK_COL].astype(int)
    cleaned_df["Positive Delay Days"] = cleaned_df["Delay Gap"].clip(lower=0)

    real_days = cleaned_df[REAL_DAYS_COL].to_numpy(dtype=float)
    scheduled_days = cleaned_df[SCHEDULED_DAYS_COL].to_numpy(dtype=float)
    adherence_ratio = np.divide(
        scheduled_days,
        real_days,
        out=np.zeros_like(scheduled_days, dtype=float),
        where=real_days > 0,
    )
    adherence_ratio = np.where(real_days > 0, np.minimum(adherence_ratio, 1.0), np.where(scheduled_days <= 0, 1.0, 0.0))
    cleaned_df["Mode Efficiency Index"] = adherence_ratio * 100

    data_quality = {
        "source_rows": int(source_rows),
        "clean_rows": int(len(cleaned_df)),
        "removed_rows": int(source_rows - len(cleaned_df)),
        "date_column": date_column,
    }
    return cleaned_df, data_quality


def summarize_dimension(df, dimension):
    summary = (
        df.groupby(dimension)
        .agg(
            Orders=(dimension, "size"),
            Avg_Actual_Days=(REAL_DAYS_COL, "mean"),
            Avg_Scheduled_Days=(SCHEDULED_DAYS_COL, "mean"),
            Avg_Delay_Gap=("Delay Gap", "mean"),
            Delayed_Days=("Positive Delay Days", "mean"),
            On_Time_Rate=("On-Time Delivery", "mean"),
            Late_Risk_Ratio=("Late Delivery Risk", "mean"),
            Efficiency_Index=("Mode Efficiency Index", "mean"),
        )
        .reset_index()
    )
    summary["On-Time Delivery Rate (%)"] = summary["On_Time_Rate"] * 100
    summary["Late Delivery Risk Ratio (%)"] = summary["Late_Risk_Ratio"] * 100
    summary["Shipping Mode Efficiency Index"] = summary["Efficiency_Index"]
    summary["Average Delivery Delay (Days)"] = summary["Avg_Delay_Gap"]
    return summary


def build_filter_sidebar(df, data_quality):
    st.sidebar.header("Filters")

    modes = sorted(df[MODE_COL].dropna().unique())
    regions = sorted(df[REGION_COL].dropna().unique())
    markets = sorted(df[MARKET_COL].dropna().unique())
    segments = sorted(df[SEGMENT_COL].dropna().unique())

    selected_modes = st.sidebar.multiselect("Shipping Mode", modes)
    selected_regions = st.sidebar.multiselect("Order Region", regions)
    selected_markets = st.sidebar.multiselect("Market", markets)
    selected_segments = st.sidebar.multiselect("Customer Segment", segments)

    filtered_df = df.copy()
    if selected_modes:
        filtered_df = filtered_df[filtered_df[MODE_COL].isin(selected_modes)]
    if selected_regions:
        filtered_df = filtered_df[filtered_df[REGION_COL].isin(selected_regions)]
    if selected_markets:
        filtered_df = filtered_df[filtered_df[MARKET_COL].isin(selected_markets)]
    if selected_segments:
        filtered_df = filtered_df[filtered_df[SEGMENT_COL].isin(selected_segments)]

    if data_quality["date_column"]:
        date_column = data_quality["date_column"]
        min_date = filtered_df[date_column].min().date()
        max_date = filtered_df[date_column].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                filtered_df[date_column].between(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                )
            ]
    else:
        st.sidebar.date_input(
            "Date Range",
            value=(date(2026, 7, 31), date(2026, 7, 31)),
            disabled=True,
        )
        st.sidebar.warning(
            "Date filter is not usable because the dataset does not have any date-related columns."
        )

    st.sidebar.metric("Validated Orders in Scope", f"{len(filtered_df):,}")
    return filtered_df


def render_header(data_quality, filtered_df):
    st.markdown(
        "<div class='page-title'>Delivery Performance, Delay Risk, and Logistics Efficiency</div>",
        unsafe_allow_html=True,
    )
    # st.caption(
    #     "Operational analytics for APL Logistics across delivery performance, shipping modes, customer segments, and regional risk."
    # )

    # if not data_quality["date_column"]:
        # st.caption(
        #     "Note: the provided dataset does not contain an order date field, so date-range analysis is not shown."
        # )

    # st.caption(f"Current filtered order count: {len(filtered_df):,}")


def render_insights(filtered_df, mode_summary, region_summary, market_summary, segment_summary):
    best_mode = mode_summary.sort_values(
        ["On-Time Delivery Rate (%)", "Average Delivery Delay (Days)"],
        ascending=[False, True],
    ).iloc[0]
    worst_region = region_summary.sort_values(
        ["Late Delivery Risk Ratio (%)", "Average Delivery Delay (Days)"],
        ascending=[False, False],
    ).iloc[0]
    worst_market = market_summary.sort_values(
        ["Late Delivery Risk Ratio (%)", "Average Delivery Delay (Days)"],
        ascending=[False, False],
    ).iloc[0]
    exposed_segment = segment_summary.sort_values(
        ["Late Delivery Risk Ratio (%)", "Orders"],
        ascending=[False, False],
    ).iloc[0]

    total_orders = len(filtered_df)
    delayed_orders = int(filtered_df["Delayed Shipment"].sum())
    delayed_share = (delayed_orders / total_orders * 100) if total_orders else 0

    st.subheader("Operational Highlights")
    cols = st.columns(4)
    insight_text = [
        (
            "Delivery Performance",
            f"{delayed_orders:,} of {total_orders:,} orders are currently delayed, representing {delayed_share:.1f}% of validated shipments.",
        ),
        (
            "Best Shipping Mode",
            f"{best_mode[MODE_COL]} has the strongest SLA performance with {best_mode['On-Time Delivery Rate (%)']:.1f}% on-time delivery.",
        ),
        (
            "Highest-Risk Geography",
            f"{worst_region[REGION_COL]} is the highest-risk region, while {worst_market[MARKET_COL]} is the highest-risk market in the current view.",
        ),
        (
            "Customer Exposure",
            f"{exposed_segment[SEGMENT_COL]} customers face the highest delay risk at {exposed_segment['Late Delivery Risk Ratio (%)']:.1f}%.",
        ),
    ]

    for column, (title, body) in zip(cols, insight_text):
        with column:
            st.markdown(
                (
                    "<div class='insight-card'>"
                    f"<div class='insight-title'>{title}</div>"
                    f"<div class='insight-body'>{body}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_overview(filtered_df, mode_summary, region_summary):
    on_time_rate = filtered_df["On-Time Delivery"].mean() * 100
    average_delay_gap = filtered_df["Delay Gap"].mean()
    late_risk_ratio = filtered_df["Late Delivery Risk"].mean() * 100
    mode_efficiency_index = filtered_df["Mode Efficiency Index"].mean()
    regional_delay_index = region_summary["Late Delivery Risk Ratio (%)"].mean()

    st.subheader("Delivery Performance Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
    k2.metric("Average Delivery Delay", f"{average_delay_gap:.2f} days")
    k3.metric("Late Delivery Risk Ratio", f"{late_risk_ratio:.1f}%")
    k4.metric("Shipping Mode Efficiency Index", f"{mode_efficiency_index:.1f}")
    k5.metric("Regional Delay Index", f"{regional_delay_index:.1f}%")

    overview_left, overview_right = st.columns([1, 1.15])

    timing_mix = (
        filtered_df["Delivery Timing"]
        .value_counts()
        .reindex(["Early", "On-time", "Delayed"], fill_value=0)
        .rename_axis("Delivery Timing")
        .reset_index(name="Orders")
    )
    fig_timing = px.pie(
        timing_mix,
        names="Delivery Timing",
        values="Orders",
        hole=0.55,
        color="Delivery Timing",
        color_discrete_map={
            "Early": "#0f766e",
            "On-time": "#2563eb",
            "Delayed": "#dc2626",
        },
        title="Delivery Timing Mix",
    )
    overview_left.plotly_chart(fig_timing, width="stretch")

    top_risk_regions = region_summary.sort_values(
        ["Late Delivery Risk Ratio (%)", "Orders"],
        ascending=[False, False],
    ).head(10)
    fig_region_risk = px.bar(
        top_risk_regions,
        x="Late Delivery Risk Ratio (%)",
        y=REGION_COL,
        orientation="h",
        title="Top Regions by Delay Risk",
        color="Average Delivery Delay (Days)",
    )
    fig_region_risk.update_layout(yaxis={"categoryorder": "total ascending"})
    overview_right.plotly_chart(fig_region_risk, width="stretch")

    st.dataframe(
        region_summary[
            [
                REGION_COL,
                "Orders",
                "On-Time Delivery Rate (%)",
                "Average Delivery Delay (Days)",
                "Late Delivery Risk Ratio (%)",
            ]
        ]
        .sort_values(["Late Delivery Risk Ratio (%)", "Orders"], ascending=[False, False]),
        hide_index=True,
        width="stretch",
    )


def render_delay_risk_dashboard(filtered_df, segment_summary):
    st.subheader("Delay Risk Analysis Dashboard")
    risk_left, risk_right = st.columns(2)

    risk_distribution = (
        filtered_df["Late Delivery Risk"]
        .map({1: "Late Delivery Risk = Yes", 0: "Late Delivery Risk = No"})
        .value_counts()
        .rename_axis("Risk Label")
        .reset_index(name="Orders")
    )
    fig_risk = px.bar(
        risk_distribution,
        x="Risk Label",
        y="Orders",
        color="Risk Label",
        title="Late Delivery Risk Distribution",
    )
    risk_left.plotly_chart(fig_risk, width="stretch")

    fig_gap = px.histogram(
        filtered_df,
        x="Delay Gap",
        color="Delivery Timing",
        nbins=40,
        title="Delivery Delay Gap Distribution",
        color_discrete_map={
            "Early": "#0f766e",
            "On-time": "#2563eb",
            "Delayed": "#dc2626",
        },
    )
    risk_right.plotly_chart(fig_gap, width="stretch")

    fig_segment = px.bar(
        segment_summary.sort_values("Late Delivery Risk Ratio (%)", ascending=False),
        x=SEGMENT_COL,
        y="Late Delivery Risk Ratio (%)",
        color="Average Delivery Delay (Days)",
        title="Delay Risk by Customer Segment",
    )
    st.plotly_chart(fig_segment, width="stretch")

    fig_status = px.box(
        filtered_df,
        x=STATUS_COL,
        y="Delay Gap",
        color=STATUS_COL,
        title="Delay Gap by Delivery Status",
    )
    st.plotly_chart(fig_status, width="stretch")


def render_shipping_mode_dashboard(filtered_df, mode_summary):
    st.subheader("Shipping Mode Comparison")
    mode_left, mode_right = st.columns(2)

    fig_mode_sla = px.bar(
        mode_summary.sort_values("On-Time Delivery Rate (%)", ascending=False),
        x=MODE_COL,
        y="On-Time Delivery Rate (%)",
        color="Shipping Mode Efficiency Index",
        title="SLA Compliance by Shipping Mode",
    )
    mode_left.plotly_chart(fig_mode_sla, width="stretch")

    fig_mode_delay = px.bar(
        mode_summary.sort_values("Average Delivery Delay (Days)", ascending=False),
        x=MODE_COL,
        y="Average Delivery Delay (Days)",
        color="Late Delivery Risk Ratio (%)",
        title="Average Delay Gap by Shipping Mode",
    )
    mode_right.plotly_chart(fig_mode_delay, width="stretch")

    timing_by_mode = (
        filtered_df.groupby([MODE_COL, "Delivery Timing"])
        .size()
        .reset_index(name="Orders")
    )
    fig_mode_mix = px.bar(
        timing_by_mode,
        x=MODE_COL,
        y="Orders",
        color="Delivery Timing",
        barmode="stack",
        title="Delivery Timing Mix by Shipping Mode",
        color_discrete_map={
            "Early": "#0f766e",
            "On-time": "#2563eb",
            "Delayed": "#dc2626",
        },
    )
    st.plotly_chart(fig_mode_mix, width="stretch")

    st.dataframe(
        mode_summary[
            [
                MODE_COL,
                "Orders",
                "On-Time Delivery Rate (%)",
                "Average Delivery Delay (Days)",
                "Late Delivery Risk Ratio (%)",
                "Shipping Mode Efficiency Index",
            ]
        ]
        .sort_values(
            ["On-Time Delivery Rate (%)", "Shipping Mode Efficiency Index"],
            ascending=[False, False],
        ),
        hide_index=True,
        width="stretch",
    )


def render_geography_dashboard(filtered_df, region_summary, market_summary):
    st.subheader("Regional and Market Diagnostics")

    geo_points = (
        filtered_df.dropna(subset=[LAT_COL, LON_COL])
        .groupby(COUNTRY_COL)
        .agg(
            Latitude=(LAT_COL, "mean"),
            Longitude=(LON_COL, "mean"),
            Orders=(COUNTRY_COL, "size"),
            Avg_Delay_Gap=("Delay Gap", "mean"),
            Late_Risk_Ratio=("Late Delivery Risk", "mean"),
        )
        .reset_index()
    )
    geo_points["Late Delivery Risk Ratio (%)"] = geo_points["Late_Risk_Ratio"] * 100

    if not geo_points.empty:
        fig_map = px.scatter_geo(
            geo_points,
            lat="Latitude",
            lon="Longitude",
            size="Orders",
            color="Late Delivery Risk Ratio (%)",
            hover_name=COUNTRY_COL,
            hover_data={
                "Orders": ":,.0f",
                "Avg_Delay_Gap": ":.2f",
                "Late Delivery Risk Ratio (%)": ":.1f",
                "Latitude": False,
                "Longitude": False,
            },
            title="Geographic Delay Visualization",
            projection="natural earth",
        )
        st.plotly_chart(fig_map, width="stretch")

    heatmap_data = (
        filtered_df.pivot_table(
            index=REGION_COL,
            columns=MARKET_COL,
            values="Late Delivery Risk",
            aggfunc="mean",
        )
        .fillna(0)
        * 100
    )
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale="Reds",
            labels={"color": "Late Delivery Risk (%)"},
            title="Regional and Market Heatmap",
        )
        st.plotly_chart(fig_heatmap, width="stretch")

    diag_left, diag_right = st.columns(2)

    fig_region = px.bar(
        region_summary.sort_values("Late Delivery Risk Ratio (%)", ascending=False).head(12),
        x="Late Delivery Risk Ratio (%)",
        y=REGION_COL,
        orientation="h",
        color="Average Delivery Delay (Days)",
        title="High-Risk Regions",
    )
    fig_region.update_layout(yaxis={"categoryorder": "total ascending"})
    diag_left.plotly_chart(fig_region, width="stretch")

    fig_market = px.bar(
        market_summary.sort_values("Shipping Mode Efficiency Index", ascending=False),
        x=MARKET_COL,
        y="Shipping Mode Efficiency Index",
        color="Late Delivery Risk Ratio (%)",
        title="Market-Wise Logistics Efficiency",
    )
    diag_right.plotly_chart(fig_market, width="stretch")

    rankings = region_summary[
        [
            REGION_COL,
            "Orders",
            "Average Delivery Delay (Days)",
            "Late Delivery Risk Ratio (%)",
        ]
    ].sort_values(
        ["Late Delivery Risk Ratio (%)", "Average Delivery Delay (Days)"],
        ascending=[False, False],
    )
    st.dataframe(rankings, hide_index=True, width="stretch")


def main():
    st.set_page_config(
        page_title="APL Logistics Delivery Performance Dashboard",
        layout="wide",
    )
    apply_theme()

    df, data_quality = load_logistics_data()
    filtered_df = build_filter_sidebar(df, data_quality)
    render_header(data_quality, filtered_df)

    if filtered_df.empty:
        st.warning("No records match the current filters. Adjust the filter selections to continue.")
        return

    mode_summary = summarize_dimension(filtered_df, MODE_COL)
    region_summary = summarize_dimension(filtered_df, REGION_COL)
    market_summary = summarize_dimension(filtered_df, MARKET_COL)
    segment_summary = summarize_dimension(filtered_df, SEGMENT_COL)

    render_insights(filtered_df, mode_summary, region_summary, market_summary, segment_summary)
    st.divider()

    overview_tab, risk_tab, mode_tab, geography_tab = st.tabs(
        [
            "Delivery Performance Overview",
            "Delay Risk Analysis",
            "Shipping Mode Comparison",
            "Regional and Market Diagnostics",
        ]
    )

    with overview_tab:
        render_overview(filtered_df, mode_summary, region_summary)

    with risk_tab:
        render_delay_risk_dashboard(filtered_df, segment_summary)

    with mode_tab:
        render_shipping_mode_dashboard(filtered_df, mode_summary)

    with geography_tab:
        render_geography_dashboard(filtered_df, region_summary, market_summary)


if __name__ == "__main__":
    main()
