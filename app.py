import streamlit as st
import pandas as pd
import numpy as np
import os

from config import *
from core.data_processing import load_dataset, validate_dataset
from core.feature_engineering import calculate_rate_of_decline
from core.scenario_engine import generate_scenario
from core.models import trend_model, hybrid_model, smoothed_model
from core.ensemble import weighted_ensemble
from core.monte_carlo import run_monte_carlo
from core.risk_engine import classify_risk

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(page_title="AquaRisk", layout="wide")

st.title("🌊 AquaRisk Dashboard")
st.subheader("Predictive Groundwater Intelligence Platform")

# ---------------------------
# LOAD DATA
# ---------------------------

df = load_dataset("data/demo_dataset.csv")
df = validate_dataset(df)

df = df.rename(columns={
    "well": "Well_ID",
    "water_level": "Groundwater_Level_m"
})

wells = df["Well_ID"].unique()

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------

st.sidebar.header("Simulation Controls")

selected_well = st.sidebar.selectbox("Select Well", wells)

selected_scenario = st.sidebar.selectbox(
    "Scenario",
    ["baseline", "drought", "expansion", "sustainable"]
)

threshold = st.sidebar.slider(
    "Critical Threshold (m)",
    min_value=5.0,
    max_value=50.0,
    value=THRESHOLD_LEVEL,
    step=0.5
)

# ---------------------------
# RUN MODEL
# ---------------------------

well_df = df[df["Well_ID"] == selected_well]
last_value = well_df["Groundwater_Level_m"].iloc[-1]
historical = well_df["Groundwater_Level_m"].values

rain, pump, et = generate_scenario(
    FORECAST_HORIZON_MONTHS,
    scenario_type=selected_scenario
)

trend_forecast = trend_model(
    last_value,
    selected_well,
    FORECAST_HORIZON_MONTHS
)

hybrid_forecast = hybrid_model(
    last_value,
    selected_well,
    FORECAST_HORIZON_MONTHS,
    rain,
    pump,
    et,
    RAIN_COEFF,
    PUMP_COEFF
)

smooth_forecast = smoothed_model(
    last_value,
    selected_well,
    FORECAST_HORIZON_MONTHS
)

ensemble_forecast = weighted_ensemble(
    hybrid_forecast,
    trend_forecast,
    smooth_forecast
)

sims, probability, p5, p95 = run_monte_carlo(
    ensemble_forecast,
    selected_well
)

risk = classify_risk(probability)

# ---------------------------
# DISPLAY RESULTS
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Exceedance Probability",
        f"{probability:.2%}"
    )

with col2:
    st.metric(
        "Risk Classification",
        risk
    )

# ---------------------------
# PLOT
# ---------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Historical
ax.plot(
    range(len(historical)),
    historical,
    label="Historical",
    linewidth=2
)

# Forecast
forecast_x = range(len(historical), len(historical) + FORECAST_HORIZON_MONTHS)

ax.plot(
    forecast_x,
    ensemble_forecast,
    label="Forecast",
    linestyle="--"
)

# Uncertainty
ax.fill_between(
    forecast_x,
    p5,
    p95,
    alpha=0.2
)

# Threshold
ax.axhline(
    threshold,
    linestyle=":",
    label="Critical Threshold"
)

ax.legend()
ax.set_ylabel("Groundwater Level (m)")
ax.set_title(f"Well {selected_well}")

st.pyplot(fig)