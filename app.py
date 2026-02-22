import streamlit as st
import pandas as pd
import numpy as np
import os

from config import *
from core.acei_engine import calculate_acei, calculate_portfolio_acei
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

st.sidebar.header("Upload Client Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload groundwater dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Client dataset loaded successfully")
else:
    df = load_dataset("data/demo_dataset.csv")
    st.sidebar.info("Using demo dataset")

# Normalizar nombres de columnas automáticamente
df.columns = df.columns.str.strip()  # elimina espacios invisibles
df.columns = df.columns.str.lower()  # todo en minúscula

df = df.rename(columns={
    "well": "Well_ID",
    "well_id": "Well_ID",
    "water_level": "Groundwater_Level_m",
    "groundwater_level_m": "Groundwater_Level_m",
    "groundwater_level": "Groundwater_Level_m"
})

# Volver a estandarizar mayúsculas finales
if "well_id" in df.columns:
    df = df.rename(columns={"well_id": "Well_ID"})
if "groundwater_level_m" in df.columns:
    df = df.rename(columns={"groundwater_level_m": "Groundwater_Level_m"})

required_columns = ["date", "Well_ID", "Groundwater_Level_m"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset must contain: date, Well_ID, Groundwater_Level_m")
    st.write("Detected columns:", df.columns)
    st.stop()
    
df = validate_dataset(df)

df = df.rename(columns={
    "well": "Well_ID",
    "water_level": "Groundwater_Level_m"
})

wells = df["Well_ID"].unique()

st.markdown("## 🛡 AquaRisk Capital Exposure Index (ACEI™)")

portfolio_scores = []

for well in wells:
    well_df = df[df["Well_ID"] == well]
    last_value = well_df["Groundwater_Level_m"].iloc[-1]
    slope = calculate_rate_of_decline(
        well_df["Groundwater_Level_m"].values
    )

    rain, pump, et = generate_scenario(
        FORECAST_HORIZON_MONTHS,
        scenario_type="baseline"
    )

    trend_forecast = trend_model(last_value, well, FORECAST_HORIZON_MONTHS)
    hybrid_forecast = hybrid_model(
        last_value, well, FORECAST_HORIZON_MONTHS,
        rain, pump, et, RAIN_COEFF, PUMP_COEFF
    )
    smooth_forecast = smoothed_model(last_value, well, FORECAST_HORIZON_MONTHS)

    ensemble_forecast = weighted_ensemble(
        hybrid_forecast,
        trend_forecast,
        smooth_forecast
    )

    sims, probability, p5, p95 = run_monte_carlo(
        ensemble_forecast,
        well
    )

    volatility = p95 - p5
    distance_to_threshold = last_value - THRESHOLD_LEVEL

    acei_score, _, _ = calculate_acei(
        probability,
        slope * 12,
        distance_to_threshold,
        volatility
    )

    portfolio_scores.append(acei_score)

portfolio_score, portfolio_category, portfolio_rec = calculate_portfolio_acei(portfolio_scores)

color = "green"
if portfolio_score > 60:
    color = "orange"
if portfolio_score > 80:
    color = "red"

st.markdown(
    f"""
    <div style="
        background-color:#111;
        padding:30px;
        border-radius:10px;
        text-align:center;
        border:2px solid {color};
    ">
        <h1 style="color:{color}; font-size:48px;">
            {portfolio_score} / 100
        </h1>
        <h3 style="color:white;">
            {portfolio_category}
        </h3>
        <p style="color:lightgray;">
            {portfolio_rec}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------

st.sidebar.header("Simulation Controls")

selected_well = st.sidebar.selectbox(
    "Select Well",
    wells
)

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
# CALCULATE INDIVIDUAL ACEI
# ---------------------------

slope = calculate_rate_of_decline(
    historical
)

volatility = p95 - p5
distance_to_threshold = last_value - threshold

acei_score, acei_category, acei_recommendation = calculate_acei(
    probability,
    slope * 12,
    distance_to_threshold,
    volatility
)

# ---------------------------
# DISPLAY RESULTS
# ---------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ACEI™️ Score",
        f"{acei_score:.2f} / 100"
    )

with col2:
    st.metric(
        "Exceedance Probability",
        f"{probability:.2%}"
    )

with col3:
    st.metric(
        "Risk Classification",
        risk
    )

st.write(f"Category: {acei_category}")
st.write(f"Advisory: {acei_recommendation}")

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