import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Nassau Candy Shipping Intelligence",
    layout="wide"
)

# -----------------------------------------------------
# CUSTOM UI STYLING
# -----------------------------------------------------
st.markdown("""
<style>

/* Sidebar tags */
span[data-baseweb="tag"]{
    background-color:#2C3E50 !important;
    color:white !important;
    border-radius:6px !important;
}

# /* Slider color */
# .stSlider > div > div > div > div{
#     background-color:#4A90E2 !important;
# }

/* KPI Card Styling */
.metric-card{
    background-color:#111827;
    padding:20px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
}

/* KPI value */
.metric-value{
    font-size:28px;
    font-weight:700;
    color:#4A90E2;
}

/* KPI label */
.metric-label{
    font-size:14px;
    color:#9CA3AF;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
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

# -----------------------------------------------------
# STATE CODE MAPPING (FOR MAP)
# -----------------------------------------------------
us_state_codes = {
"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR",
"California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE",
"Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
"Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS",
"Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
"Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
"Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
"New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
"North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
"Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
"South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT",
"Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
"Wisconsin":"WI","Wyoming":"WY"
}

df["State Code"] = df["State/Province"].map(us_state_codes)

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.markdown("### 🔧 Filters")

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
    1,15,5
)

# -----------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------
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

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.title("📦 Nassau Candy Distributor — Shipping Intelligence Dashboard")

st.divider()

# -----------------------------------------------------
# KPI CALCULATIONS
# -----------------------------------------------------
avg_lead_time = filtered_df["Shipping Lead Time"].mean()

route_volume = filtered_df.shape[0]

delay_frequency = (
    (filtered_df["Shipping Lead Time"] > delay_threshold).mean()*100
)

route_perf = filtered_df.groupby("Route").agg(
    avg_lead_time=("Shipping Lead Time","mean"),
    shipments=("Order ID","count")
).reset_index()

route_perf["Efficiency Score"] = 1 - (
(route_perf["avg_lead_time"]-route_perf["avg_lead_time"].min())/
(route_perf["avg_lead_time"].max()-route_perf["avg_lead_time"].min())
)

efficiency_score = route_perf["Efficiency Score"].mean()

# -----------------------------------------------------
# KPI DISPLAY (IMPROVED UI)
# -----------------------------------------------------
c1,c2,c3,c4,c5 = st.columns(5)

c1.markdown(f"""
<div class="metric-card">
<div class="metric-value">{avg_lead_time:.1f}</div>
<div class="metric-label">Shipping Lead Time</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card">
<div class="metric-value">{route_perf['avg_lead_time'].mean():.1f}</div>
<div class="metric-label">Average Lead Time</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card">
<div class="metric-value">{route_volume}</div>
<div class="metric-label">Route Volume</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric-card">
<div class="metric-value">{delay_frequency:.1f}%</div>
<div class="metric-label">Delay Frequency</div>
</div>
""", unsafe_allow_html=True)

c5.markdown(f"""
<div class="metric-card">
<div class="metric-value">{efficiency_score:.2f}</div>
<div class="metric-label">Route Efficiency Score</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------
# ROUTE EFFICIENCY
# -----------------------------------------------------
st.subheader("🚚 Route Efficiency Overview")

top_routes = route_perf.sort_values("avg_lead_time").head(10)
slow_routes = route_perf.sort_values(
    "avg_lead_time",ascending=False
).head(10)

col1,col2 = st.columns(2)

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

col1.plotly_chart(fig1,use_container_width=True)
col2.plotly_chart(fig2,use_container_width=True)

st.divider()

# -----------------------------------------------------
# SHIPPING MAP
# -----------------------------------------------------
st.subheader("🗺 US Shipping Efficiency Map")

state_perf = filtered_df.groupby("State Code").agg(
    avg_lead_time=("Shipping Lead Time","mean")
).reset_index()

fig_map = px.choropleth(
    state_perf,
    locations="State Code",
    locationmode="USA-states",
    color="avg_lead_time",
    scope="usa"
)

st.plotly_chart(fig_map,use_container_width=True)

st.divider()

# -----------------------------------------------------
# SHIP MODE PERFORMANCE
# -----------------------------------------------------
st.subheader("🚛 Ship Mode Performance")

ship_perf = filtered_df.groupby("Ship Mode").agg(
    avg_lead_time=("Shipping Lead Time","mean"),
    shipments=("Order ID","count")
).reset_index()

fig_ship = px.bar(
    ship_perf,
    x="Ship Mode",
    y="avg_lead_time",
    text="avg_lead_time"
)

st.plotly_chart(fig_ship,use_container_width=True)

st.divider()

# -----------------------------------------------------
# LEAD TIME DISTRIBUTION
# -----------------------------------------------------
st.subheader("📊 Shipping Lead Time Distribution")

fig_hist = px.histogram(
    filtered_df,
    x="Shipping Lead Time",
    nbins=40
)

st.plotly_chart(fig_hist,use_container_width=True)

st.divider()

# -----------------------------------------------------
# ROUTE VOLUME VS LEAD TIME
# -----------------------------------------------------
st.subheader("📦 Route Volume vs Lead Time")

fig_scatter = px.scatter(
    route_perf,
    x="shipments",
    y="avg_lead_time",
    size="shipments",
    hover_data=["Route"]
)

st.plotly_chart(fig_scatter,use_container_width=True)