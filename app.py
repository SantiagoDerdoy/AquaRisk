# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from config import (
    THRESHOLD_LEVEL,
    FORECAST_HORIZON_MONTHS,
    RAIN_COEFF,
    PUMP_COEFF,
    DECLINE_RATES
)

from core.models import trend_model, hybrid_model, smoothed_model
from core.monte_carlo import run_monte_carlo
from core.scenario_engine import generate_scenario
from core.ensemble import weighted_ensemble
from core.risk_engine import classify_risk, calculate_time_to_threshold
from core.acei_engine import calculate_acei
from core.report_generator import generate_executive_pdf


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AquaRisk | Groundwater Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use("default")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] {
        background-color: #080e1a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1526;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Typography */
    h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.5px; }
    h1 { font-size: 2.1rem !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0f1c30;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.5rem !important; font-weight: 700 !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Sidebar labels */
    .css-1d391kg, [data-testid="stSidebarContent"] label { color: #94a3b8 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(37,99,235,0.35);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #065f46, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown(
        "<h1 style='color:#f1f5f9; margin-bottom:0;'>AquaRisk</h1>"
        "<p style='color:#64748b; margin-top:2px; font-size:0.95rem; letter-spacing:0.04em;'>"
        "PREDICTIVE GROUNDWATER INTELLIGENCE PLATFORM</p>",
        unsafe_allow_html=True
    )

st.markdown("---")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

st.sidebar.markdown("## Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload groundwater dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/demo_dataset.csv")

df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    "well": "Well_ID",
    "well_id": "Well_ID",
    "water_level": "Groundwater_Level_m",
    "groundwater_level": "Groundwater_Level_m",
    "groundwater_level_m": "Groundwater_Level_m"
})

wells = df["Well_ID"].unique()


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.markdown("## Simulation")

selected_well = st.sidebar.selectbox("Select Well", wells)

selected_scenario = st.sidebar.selectbox(
    "Scenario",
    ["baseline", "drought", "expansion", "sustainable"],
    format_func=lambda x: {
        "baseline": "Baseline Operating",
        "drought": "Climatic Stress",
        "expansion": "Production Expansion",
        "sustainable": "Sustainability Strategy"
    }[x]
)

threshold = st.sidebar.slider(
    "Critical Threshold (m)",
    5.0, 50.0,
    float(THRESHOLD_LEVEL)
)


# --------------------------------------------------
# PORTFOLIO ACEI (runs on all wells, baseline scenario)
# --------------------------------------------------

portfolio_scores = []

for well in wells:
    well_df_port = df[df["Well_ID"] == well]
    historical_port = well_df_port["Groundwater_Level_m"].values
    last_val_port = float(historical_port[-1])

    if len(historical_port) < 3:
        slope_port = -DECLINE_RATES.get(well, 0.5) / 12
    else:
        x_port = np.arange(len(historical_port))
        slope_port, _ = np.polyfit(x_port, historical_port, 1)

    rain_p, pump_p, et_p = generate_scenario(FORECAST_HORIZON_MONTHS, "baseline")

    t_p = trend_model(last_val_port, slope_port, FORECAST_HORIZON_MONTHS)
    h_p = hybrid_model(last_val_port, slope_port, FORECAST_HORIZON_MONTHS,
                       rain_p, pump_p, et_p, RAIN_COEFF[well], PUMP_COEFF[well])
    s_p = smoothed_model(last_val_port, slope_port, FORECAST_HORIZON_MONTHS)
    ens_p = weighted_ensemble(h_p, t_p, s_p)

    _, prob_p, p5_p, p95_p = run_monte_carlo(ens_p, well)

    volatility_p = float(np.mean(p95_p - p5_p))
    distance_p = last_val_port - threshold

    acei_p, _, _ = calculate_acei(
        float(np.asarray(prob_p).mean()),
        slope_port * 12,
        distance_p,
        volatility_p
    )
    portfolio_scores.append(acei_p)

portfolio_score = float(np.mean(portfolio_scores))

if portfolio_score < 40:
    port_cat = "Low Exposure"
    port_color = "#10b981"
elif portfolio_score < 70:
    port_cat = "Moderate Exposure"
    port_color = "#f59e0b"
else:
    port_cat = "High Exposure"
    port_color = "#ef4444"


# --------------------------------------------------
# PORTFOLIO ACEI DISPLAY
# --------------------------------------------------

st.markdown("### Portfolio Risk Overview")

pcol1, pcol2, pcol3 = st.columns([2, 1, 1])

with pcol1:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f1c30, #0d1a2e);
            border: 1px solid {port_color}44;
            border-left: 4px solid {port_color};
            border-radius: 12px;
            padding: 28px 32px;
            display: flex;
            align-items: center;
            gap: 24px;
        ">
            <div>
                <div style="color:#94a3b8; font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">
                    Portfolio ACEI™
                </div>
                <div style="color:{port_color}; font-size:3rem; font-weight:800; line-height:1;">
                    {portfolio_score:.1f}
                    <span style="font-size:1.2rem; color:#64748b;">/ 100</span>
                </div>
                <div style="color:#e2e8f0; font-size:1rem; font-weight:600; margin-top:6px;">
                    {port_cat}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with pcol2:
    st.metric("Wells Monitored", len(wells))

with pcol3:
    st.metric("Forecast Horizon", f"{FORECAST_HORIZON_MONTHS} months")

st.markdown("---")


# --------------------------------------------------
# WELL DATA
# --------------------------------------------------

well_df = df[df["Well_ID"] == selected_well]
historical = well_df["Groundwater_Level_m"].values
last_value = float(historical[-1])

if len(historical) < 3:
    slope = -DECLINE_RATES.get(selected_well, 0.5) / 12
    st.sidebar.warning("Insufficient historical data. Using fallback decline rate.")
else:
    x = np.arange(len(historical))
    slope, _ = np.polyfit(x, historical, 1)


# --------------------------------------------------
# SCENARIO & MODELS
# --------------------------------------------------

rain, pump, et = generate_scenario(FORECAST_HORIZON_MONTHS, selected_scenario)

trend_forecast = trend_model(last_value, slope, FORECAST_HORIZON_MONTHS)
hybrid_forecast = hybrid_model(
    last_value, slope, FORECAST_HORIZON_MONTHS,
    rain, pump, et,
    RAIN_COEFF[selected_well], PUMP_COEFF[selected_well]
)
smooth_forecast = smoothed_model(last_value, slope, FORECAST_HORIZON_MONTHS)
ensemble_forecast = weighted_ensemble(hybrid_forecast, trend_forecast, smooth_forecast)


# --------------------------------------------------
# MONTE CARLO
# --------------------------------------------------

sims, probability, p5, p95 = run_monte_carlo(ensemble_forecast, selected_well)
probability = float(np.asarray(probability).mean())
risk = classify_risk(probability)

# Ensure continuity at forecast start
ensemble_with_start = np.insert(ensemble_forecast, 0, last_value)
p5_plot = np.insert(p5, 0, last_value)
p95_plot = np.insert(p95, 0, last_value)


# --------------------------------------------------
# TIME TO THRESHOLD  (using risk_engine)
# --------------------------------------------------

mean_cross, prob_12, prob_24 = calculate_time_to_threshold(sims, threshold)


# --------------------------------------------------
# ACEI
# --------------------------------------------------

volatility = float(np.mean(p95 - p5))
distance_to_threshold = last_value - threshold

acei_score, acei_category, acei_recommendation = calculate_acei(
    probability,
    slope * 12,
    distance_to_threshold,
    volatility
)


# --------------------------------------------------
# WELL METRICS
# --------------------------------------------------

st.markdown(f"### Well {selected_well} — {selected_scenario.capitalize()} Scenario")

m1, m2, m3, m4 = st.columns(4)
m1.metric("ACEI™ Score", f"{acei_score:.1f} / 100", help="Asset Capital Exposure Index")
m2.metric("Exceedance Probability", f"{probability:.1%}")
m3.metric("Risk Classification", risk)

if mean_cross is not None:
    m4.metric("Expected Threshold Crossing", f"{mean_cross:.1f} mo")
else:
    m4.metric("Expected Threshold Crossing", "Not reached")

# ACEI advisory
acei_color = "#10b981" if acei_score < 40 else "#f59e0b" if acei_score < 70 else "#ef4444"
st.markdown(
    f"""
    <div style="
        background:#0f1c30;
        border-left: 4px solid {acei_color};
        border-radius: 8px;
        padding: 14px 20px;
        margin: 12px 0;
        color:#cbd5e1;
        font-size:0.9rem;
    ">
        <span style="color:{acei_color}; font-weight:700;">{acei_category}</span>
        &nbsp;·&nbsp; {acei_recommendation}
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# THRESHOLD PROBABILITY METRICS
# --------------------------------------------------

if mean_cross is not None:
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("P(breach within 12 mo)", f"{prob_12:.1%}")
    tc2.metric("P(breach within 24 mo)", f"{prob_24:.1%}")
    tc3.metric("Mean Crossing Month", f"{mean_cross:.1f}")

st.markdown("---")


# --------------------------------------------------
# FORECAST PLOT
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#080e1a")
ax.set_facecolor("#0d1526")

x_hist = np.arange(len(historical))
x_fore = np.arange(len(historical) - 1, len(historical) - 1 + len(ensemble_with_start))

ax.plot(x_hist, historical,
        color="#60a5fa", linewidth=2.5, label="Historical", zorder=3)

ax.plot(x_fore, ensemble_with_start,
        color="#f59e0b", linewidth=2, linestyle="--", label="Forecast (Mean)", zorder=3)

ax.fill_between(x_fore, p5_plot, p95_plot,
                color="#f59e0b", alpha=0.12, label="Uncertainty P5–P95")

ax.axhline(threshold,
           color="#ef4444", linewidth=1.5, linestyle="--",
           label=f"Critical Threshold ({threshold} m)", zorder=2)

ax.axvline(len(historical) - 1,
           color="#475569", linewidth=1, linestyle=":", label="Forecast Start")

ax.set_title(
    f"Well {selected_well}  ·  {selected_scenario.title()} Scenario",
    color="#e2e8f0", fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Months", color="#94a3b8", fontsize=10)
ax.set_ylabel("Groundwater Level (m)", color="#94a3b8", fontsize=10)
ax.tick_params(colors="#64748b")
ax.spines[:].set_color("#1e293b")
ax.grid(True, linestyle="--", alpha=0.15, color="#334155")
ax.legend(facecolor="#0d1526", edgecolor="#1e293b", labelcolor="#cbd5e1", fontsize=9)

st.pyplot(fig)


# --------------------------------------------------
# SAVE CHART FOR PDF
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)
chart_path = f"reports/forecast_{selected_well}.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)


# --------------------------------------------------
# MONTE CARLO DISTRIBUTION
# --------------------------------------------------

st.markdown("#### Monte Carlo — Final Level Distribution")

fig_mc, ax_mc = plt.subplots(figsize=(9, 3.5))
fig_mc.patch.set_facecolor("#080e1a")
ax_mc.set_facecolor("#0d1526")

final_values = sims[:, -1]
ax_mc.hist(final_values, bins=40, color="#2563eb", alpha=0.75, edgecolor="none")
ax_mc.axvline(threshold, color="#ef4444", linewidth=2, linestyle="--",
              label=f"Threshold ({threshold} m)")
ax_mc.axvline(float(np.mean(final_values)), color="#f59e0b", linewidth=1.5,
              linestyle="--", label=f"Mean ({np.mean(final_values):.1f} m)")

ax_mc.set_title("Distribution of Simulated Final Water Levels",
                color="#e2e8f0", fontsize=11, pad=10)
ax_mc.set_xlabel("Water Level (m)", color="#94a3b8", fontsize=9)
ax_mc.set_ylabel("Frequency", color="#94a3b8", fontsize=9)
ax_mc.tick_params(colors="#64748b")
ax_mc.spines[:].set_color("#1e293b")
ax_mc.grid(True, linestyle="--", alpha=0.15, color="#334155")
ax_mc.legend(facecolor="#0d1526", edgecolor="#1e293b", labelcolor="#cbd5e1", fontsize=9)

st.pyplot(fig_mc)
plt.close(fig_mc)

st.markdown("---")


# --------------------------------------------------
# EXECUTIVE PDF  (single button, no double-download)
# --------------------------------------------------

st.markdown("### Executive Report")

if st.button("Generate Executive PDF Report"):

    pdf_bytes = generate_executive_pdf(
        well_name=selected_well,
        acei_score=acei_score,
        acei_category=acei_category,
        exceedance_probability=probability,
        risk_classification=risk,
        time_to_threshold=int(mean_cross) if mean_cross is not None else None,
        final_forecast_value=float(ensemble_forecast[-1]),
        threshold_value=threshold,
        chart_path=chart_path
    )

    st.download_button(
        label="⬇ Download Executive Report (PDF)",
        data=pdf_bytes,
        file_name=f"AquaRisk_Report_{selected_well}.pdf",
        mime="application/pdf"
    )