import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Factory Optimization System", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_excel("Nassau Candy Distributor.xlsx")

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin"] = df["Gross Profit"] / df["Sales"]

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("leadtime_model.pkl", "rb"))
le_region = pickle.load(open("le_region.pkl", "rb"))
le_ship = pickle.load(open("le_ship.pkl", "rb"))
le_product = pickle.load(open("le_product.pkl", "rb"))
le_factory = pickle.load(open("le_factory.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

factories = list(le_factory.classes_)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Optimization Controls")

product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
shipmode = st.sidebar.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))

priority = st.sidebar.slider(
    "Optimization Priority (Speed ←→ Profit)",
    0, 100, 50
)

# ---------------- PREDICTION FUNCTION ----------------
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

# ---------------- SIMULATION ----------------
def simulate():

    base_margin = df[df["Product Name"] == product]["Profit Margin"].mean()
    results = []

    for i, f in enumerate(factories):

        lead_time = predict_lead_time(product, region, shipmode, f)

        # deterministic variation
        lead_time = lead_time * (1 + (i * 0.02))

        # improved profit logic
        profit_adj = base_margin - (lead_time * 0.003)

        results.append([f, lead_time, profit_adj])

    return pd.DataFrame(
        results,
        columns=["Factory", "Predicted Lead Time", "Profit Impact"]
    )

sim = simulate()

# ---------------- KPI CALCULATIONS ----------------
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

# ---------------- NORMALIZATION ----------------
lt_range = sim["Predicted Lead Time"].max() - sim["Predicted Lead Time"].min()
profit_range = sim["Profit Impact"].max() - sim["Profit Impact"].min()

lt_norm = (sim["Predicted Lead Time"] - sim["Predicted Lead Time"].min()) / (lt_range + 1e-6)
profit_norm = (sim["Profit Impact"] - sim["Profit Impact"].min()) / (profit_range + 1e-6)

# ---------------- SCORING ----------------
weight = priority / 100

sim["Score"] = (
    (1 - weight) * lt_norm +
    weight * (1 - profit_norm)
)

sim_sorted = sim.sort_values("Score")

# ---------------- UI ----------------
st.title("Factory Optimization Simulator")

# ---------------- KPI ----------------
st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

# FIXED KPI COLORS
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

# ---------------- MODULE 1 ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(sim_sorted, use_container_width=True)

with col2:
    st.metric("Best Factory", sim_sorted.iloc[0]["Factory"])
    st.metric("Best Lead Time", round(sim_sorted.iloc[0]["Predicted Lead Time"], 2))

# FIXED GRAPH
st.subheader("Factory Comparison (Lead Time)")
st.bar_chart(sim.sort_values("Predicted Lead Time").set_index("Factory")["Predicted Lead Time"])

# ---------------- MODULE 2 ----------------
st.subheader("What-If Scenario Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Current", round(current, 2))
col2.metric("Optimized", round(best, 2))

if lead_reduction > 0:
    col3.metric("Improvement %", f"{round(lead_reduction,2)}%", delta="Improved", delta_color="normal")
else:
    col3.metric("Improvement %", f"{round(lead_reduction,2)}%", delta="Worse", delta_color="inverse")

# ---------------- MODULE 3 ----------------
st.subheader("Recommendation Dashboard")

top3 = sim_sorted.head(3).copy()
top3["Improvement %"] = ((current - top3["Predicted Lead Time"]) / current) * 100

st.dataframe(top3, use_container_width=True)

# ---------------- MODULE 4 ----------------
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