# app.py
# ─────────────────────────────────────────────
# AquaRisk — Main Streamlit Application
# Includes: Auth, Plan gating, Well management,
#           Data entry, Risk dashboard
# ─────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import date, timedelta

from config import (
    THRESHOLD_LEVEL,
    FORECAST_HORIZON_MONTHS,
    RAIN_COEFF,
    PUMP_COEFF,
    DECLINE_RATES,
    NOISE_SD,
)

from core.auth import (
    login_user,
    logout_user,
    register_user,
    get_user_wells,
    create_well,
    delete_well,
    get_well_readings,
    save_readings_bulk,
    log_analysis_run,
    PLAN_LIMITS,
)
from core.models import trend_model, hybrid_model, smoothed_model
from core.monte_carlo import run_monte_carlo
from core.scenario_engine import generate_scenario
from core.ensemble import weighted_ensemble
from core.risk_engine import classify_risk, calculate_time_to_threshold
from core.acei_engine import calculate_acei
from core.report_generator import generate_executive_pdf
from core.weather import get_real_rain_series, apply_scenario_modifier, get_weather_summary


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AquaRisk | Groundwater Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use("default")


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #060d1a;
    color: #cbd5e1;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0a1628;
    border-right: 1px solid rgba(255,255,255,0.06);
}
h1 { font-family: 'DM Serif Display', serif; font-size: 2rem !important; color: #f1f5f9 !important; }
h2 { font-family: 'DM Serif Display', serif; color: #e2e8f0 !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 1rem !important; }

[data-testid="stMetric"] {
    background: #0e1e35;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-size: 1.4rem !important; font-weight: 700 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 24px; width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    box-shadow: 0 8px 20px rgba(37,99,235,0.35);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #059669) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; width: 100%;
}
.stTextInput > div > input,
.stSelectbox > div > div,
.stNumberInput > div > input {
    background: #0e1e35 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }
div[data-testid="stExpander"] {
    background: #0e1e35;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_session():
    defaults = {
        "user": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def plan_badge(plan: str) -> str:
    colors = {"professional": "#475569", "advanced": "#1d4ed8", "enterprise": "#d4a853"}
    labels = {"professional": "Professional", "advanced": "Advanced", "enterprise": "Enterprise"}
    c = colors.get(plan, "#475569")
    l = labels.get(plan, plan.title())
    return (
        f'<span style="background:{c}; color:white; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.08em; text-transform:uppercase; padding:4px 10px; '
        f'border-radius:100px;">{l}</span>'
    )


def locked_feature(name: str, required_plan: str = "Advanced"):
    st.markdown(
        f"""<div style="background:#0e1e35; border:1px solid rgba(255,255,255,0.07);
            border-radius:10px; padding:24px; text-align:center; color:#475569;">
            🔒 <strong style="color:#64748b;">{name}</strong>
            <div style="font-size:0.8rem; margin-top:6px;">Available on {required_plan} plan and above.</div>
        </div>""",
        unsafe_allow_html=True
    )


def get_noise_sd(well_name: str) -> float:
    return NOISE_SD.get(well_name, 0.3)


def patch_noise_sd(well_name: str):
    """Add well to NOISE_SD config if missing, return cleanup function."""
    import config as _cfg
    added = well_name not in _cfg.NOISE_SD
    if added:
        _cfg.NOISE_SD[well_name] = 0.3
    def cleanup():
        if added and well_name in _cfg.NOISE_SD:
            del _cfg.NOISE_SD[well_name]
    return cleanup


# ─────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────

def render_auth():
    st.markdown(
        """<div style="text-align:center; padding:60px 0 20px;">
            <div style="font-family:'DM Serif Display',serif; font-size:2.8rem; color:#f1f5f9;">AquaRisk</div>
            <div style="color:#64748b; font-size:0.85rem; letter-spacing:0.12em; text-transform:uppercase; margin-top:6px;">
                Predictive Groundwater Intelligence
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    col = st.columns([1, 1.4, 1])[1]

    with col:
        tabs = st.tabs(["Sign In", "Create Account"])

        # LOGIN
        with tabs[0]:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                email    = st.text_input("Email address")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    with st.spinner("Authenticating..."):
                        session, err = login_user(email.strip(), password)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.user = session
                        st.rerun()

        # REGISTER
        with tabs[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("register_form"):
                r_name    = st.text_input("Full name")
                r_company = st.text_input("Company / Operation name")
                r_email   = st.text_input("Email address")
                r_pass    = st.text_input("Password (min. 6 characters)", type="password")
                r_pass2   = st.text_input("Confirm password", type="password")
                r_plan    = st.selectbox(
                    "Select plan",
                    options=["professional", "advanced", "enterprise"],
                    format_func=lambda x: {
                        "professional": "Professional — $499/mo · 2 wells · min. 3 months",
                        "advanced":     "Advanced — $990/mo · 4 wells · min. 6 months",
                        "enterprise":   "Enterprise — From $1,800/mo · 5 wells · min. 12 months",
                    }[x]
                )
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#64748b; margin-top:4px;'>"
                    f"Minimum commitment: <strong style='color:#94a3b8;'>"
                    f"{PLAN_LIMITS[r_plan]['min_months']} months</strong>.</div>",
                    unsafe_allow_html=True
                )
                submitted = st.form_submit_button("Create Account", use_container_width=True)

            if submitted:
                errors = []
                if not r_name.strip():    errors.append("Full name is required.")
                if not r_company.strip(): errors.append("Company name is required.")
                if not r_email.strip():   errors.append("Email is required.")
                if len(r_pass) < 6:       errors.append("Password must be at least 6 characters.")
                if r_pass != r_pass2:     errors.append("Passwords do not match.")
                if errors:
                    for e in errors: st.error(e)
                else:
                    with st.spinner("Creating your account..."):
                        session, err = register_user(
                            email=r_email.strip(), password=r_pass,
                            full_name=r_name.strip(), company=r_company.strip(), plan=r_plan,
                        )
                    if err and "confirm" in err.lower():
                        st.success(err)
                    elif err:
                        st.error(err)
                    else:
                        st.session_state.user = session
                        st.rerun()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(user):
    with st.sidebar:
        st.markdown(
            f"""<div style="padding:16px 0 12px;">
                <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#f1f5f9;">AquaRisk</div>
                <div style="margin-top:6px;">{plan_badge(user.plan)}</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='font-size:0.82rem; color:#64748b; padding-bottom:12px;'>"
            f"{user.full_name}<br><span style='color:#475569;'>{user.company}</span></div>",
            unsafe_allow_html=True
        )

        if user.days_remaining is not None:
            days  = user.days_remaining
            color = "#10b981" if days > 30 else "#f59e0b" if days > 7 else "#ef4444"
            st.markdown(
                f"<div style='font-size:0.78rem; color:{color}; margin-bottom:12px;'>"
                f"Plan expires in {days} days</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        page = st.radio("Navigation", ["Dashboard", "My Wells", "Data Entry", "Account"],
                        label_visibility="collapsed")
        st.markdown("---")

        selected_scenario = None
        threshold = float(THRESHOLD_LEVEL)

        if page == "Dashboard":
            st.markdown("**Simulation**")
            selected_scenario = st.selectbox(
                "Scenario",
                ["baseline", "drought", "expansion", "sustainable"],
                format_func=lambda x: {
                    "baseline": "Baseline Operating", "drought": "Climatic Stress",
                    "expansion": "Production Expansion", "sustainable": "Sustainability Strategy",
                }[x],
                label_visibility="collapsed"
            )
            threshold = st.slider("Critical Threshold (m)", 5.0, 50.0, float(THRESHOLD_LEVEL))
            st.markdown("---")

        if st.button("Sign Out"):
            logout_user()
            st.session_state.user = None
            st.rerun()

    return page, selected_scenario, threshold


# ─────────────────────────────────────────────
# PAGE: MY WELLS
# ─────────────────────────────────────────────

def render_wells_page(user):
    st.title("My Wells")
    st.markdown(
        f"<div style='color:#64748b; margin-bottom:24px;'>Your plan allows "
        f"<strong style='color:#94a3b8;'>{user.max_wells} well(s)</strong>.</div>",
        unsafe_allow_html=True
    )

    wells = get_user_wells(user.user_id)

    with st.expander("➕ Add New Well", expanded=len(wells) == 0):
        with st.form("add_well_form"):
            col1, col2 = st.columns(2)
            with col1: well_name = st.text_input("Well ID / Name", placeholder="e.g. GW-01")
            with col2: well_loc  = st.text_input("Location / Field", placeholder="e.g. North Field")
            submitted = st.form_submit_button("Add Well")
        if submitted:
            if not well_name.strip():
                st.error("Well name is required.")
            else:
                well, err = create_well(user.user_id, well_name.strip(), well_loc.strip(), user.max_wells)
                if err: st.error(err)
                else:
                    st.success(f"Well **{well['well_name']}** added.")
                    st.rerun()

    if not wells:
        st.info("No wells yet. Add your first well above.")
        return

    st.markdown(f"**{len(wells)} well(s) registered**")

    for w in wells:
        c1, c2, c3 = st.columns([3, 3, 1])
        with c1:
            st.markdown(
                f"<div style='color:#e2e8f0; font-weight:600;'>{w['well_name']}</div>"
                f"<div style='color:#64748b; font-size:0.82rem;'>{w.get('location','—')}</div>",
                unsafe_allow_html=True
            )
        with c2:
            readings = get_well_readings(w["id"], user.user_id)
            st.markdown(
                f"<div style='color:#94a3b8; font-size:0.85rem;'>{len(readings)} reading(s)</div>",
                unsafe_allow_html=True
            )
        with c3:
            if st.button("Delete", key=f"del_{w['id']}"):
                err = delete_well(w["id"], user.user_id)
                if err: st.error(err)
                else: st.rerun()
        st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,0.05); margin:10px 0;'></div>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DATA ENTRY
# ─────────────────────────────────────────────

def render_data_entry(user):
    st.title("Data Entry")
    st.markdown(
        "<div style='color:#64748b; margin-bottom:24px;'>"
        "Enter groundwater level readings manually or upload a CSV file.</div>",
        unsafe_allow_html=True
    )

    wells = get_user_wells(user.user_id)
    if not wells:
        st.warning("No wells found. Go to **My Wells** first.")
        return

    well_options = {w["well_name"]: w for w in wells}
    selected_well_name = st.selectbox("Select Well", list(well_options.keys()))
    well_id = well_options[selected_well_name]["id"]
    existing = get_well_readings(well_id, user.user_id)

    tab1, tab2 = st.tabs(["Manual Entry", "Upload CSV"])

    # MANUAL ENTRY
    with tab1:
        st.markdown(
            "<div style='font-size:0.85rem; color:#64748b; margin-bottom:16px;'>"
            "One reading per row. Date format: YYYY-MM-DD. Water level in meters.</div>",
            unsafe_allow_html=True
        )

        if existing:
            init_data = pd.DataFrame([
                {"Date": r["reading_date"], "Water Level (m)": float(r["water_level"]),
                 "Notes": r.get("notes", "")}
                for r in existing
            ])
        else:
            today = date.today()
            init_data = pd.DataFrame([
                {"Date": str(today - timedelta(days=30*i)), "Water Level (m)": 0.0, "Notes": ""}
                for i in range(5, -1, -1)
            ])

        edited_df = st.data_editor(
            init_data, num_rows="dynamic", use_container_width=True,
            column_config={
                "Date": st.column_config.TextColumn("Date (YYYY-MM-DD)", width="medium"),
                "Water Level (m)": st.column_config.NumberColumn(
                    "Water Level (m)", min_value=0.0, max_value=500.0, step=0.01, width="medium"),
                "Notes": st.column_config.TextColumn("Notes (optional)", width="large"),
            },
            key=f"editor_{well_id}"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("💾 Save Readings"):
                rows, errors = [], []
                for i, row in edited_df.iterrows():
                    date_val = str(row["Date"]).strip()
                    try:
                        date.fromisoformat(date_val)
                    except ValueError:
                        errors.append(f"Row {i+1}: invalid date '{date_val}'. Use YYYY-MM-DD.")
                        continue
                    try:
                        level = float(row["Water Level (m)"])
                        if not (0 <= level <= 500):
                            errors.append(f"Row {i+1}: level must be 0–500 m.")
                            continue
                    except (TypeError, ValueError):
                        errors.append(f"Row {i+1}: water level must be a number.")
                        continue
                    rows.append({"reading_date": date_val, "water_level": level,
                                 "notes": str(row.get("Notes","")).strip()})

                if errors:
                    for e in errors: st.error(e)
                    st.warning("⚠️ Readings not saved. Fix errors above — no analysis credits consumed.")
                elif len(rows) < 3:
                    st.error("Please enter at least 3 readings to enable forecasting.")
                else:
                    err = save_readings_bulk(well_id, user.user_id, rows)
                    if err: st.error(err)
                    else:
                        st.success(f"✅ {len(rows)} readings saved for **{selected_well_name}**.")
                        st.rerun()

        with col2:
            if existing:
                st.markdown(
                    f"<div style='color:#64748b; font-size:0.82rem; padding-top:12px;'>"
                    f"Saved: {len(existing)} readings · Last: {existing[-1]['reading_date']}</div>",
                    unsafe_allow_html=True
                )

    # CSV UPLOAD
    with tab2:
        st.markdown(
            "<div style='font-size:0.85rem; color:#64748b; margin-bottom:16px;'>"
            "CSV must have columns: <code>date</code> (YYYY-MM-DD) and "
            "<code>water_level</code> (numeric). Optional: <code>notes</code>.</div>",
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
                csv_df.columns = csv_df.columns.str.strip().str.lower()
                missing = {"date", "water_level"} - set(csv_df.columns)
                if missing:
                    st.error(f"CSV missing columns: {missing}")
                else:
                    csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(csv_df.head(10), use_container_width=True)
                    if st.button("Import CSV Readings"):
                        rows = [{"reading_date": r["date"], "water_level": float(r["water_level"]),
                                 "notes": str(r.get("notes",""))} for _, r in csv_df.iterrows()]
                        err = save_readings_bulk(well_id, user.user_id, rows)
                        if err: st.error(err)
                        else:
                            st.success(f"✅ {len(rows)} readings imported for **{selected_well_name}**.")
                            st.rerun()
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.warning("⚠️ File not imported. No analysis credits consumed.")


# ─────────────────────────────────────────────
# PAGE: ACCOUNT
# ─────────────────────────────────────────────

def render_account(user):
    st.title("Account")
    limits = user.limits

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Account Details**")
        st.markdown(
            f"""<div style="background:#0e1e35; border:1px solid rgba(255,255,255,0.07);
                border-radius:10px; padding:20px; font-size:0.9rem; color:#94a3b8;">
                <div style="margin-bottom:10px;"><span style="color:#64748b;">Name</span><br>
                    <strong style="color:#e2e8f0;">{user.full_name}</strong></div>
                <div style="margin-bottom:10px;"><span style="color:#64748b;">Company</span><br>
                    <strong style="color:#e2e8f0;">{user.company}</strong></div>
                <div><span style="color:#64748b;">Email</span><br>
                    <strong style="color:#e2e8f0;">{user.email}</strong></div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("**Plan Details**")
        exp_str  = user.plan_expires_at.strftime("%B %d, %Y") if user.plan_expires_at else "—"
        days     = user.days_remaining
        days_str = f"{days} days remaining" if days is not None else "—"
        color    = "#10b981" if (days or 999) > 30 else "#f59e0b" if (days or 999) > 7 else "#ef4444"
        st.markdown(
            f"""<div style="background:#0e1e35; border:1px solid rgba(255,255,255,0.07);
                border-radius:10px; padding:20px; font-size:0.9rem; color:#94a3b8;">
                <div style="margin-bottom:10px;"><span style="color:#64748b;">Plan</span><br>
                    {plan_badge(user.plan)}</div>
                <div style="margin-bottom:10px;"><span style="color:#64748b;">Status</span><br>
                    <strong style="color:#10b981;">{user.plan_status.title()}</strong></div>
                <div style="margin-bottom:10px;"><span style="color:#64748b;">Expires</span><br>
                    <strong style="color:#e2e8f0;">{exp_str}</strong></div>
                <div><strong style="color:{color};">{days_str}</strong></div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Plan Features**")
    feat_df = pd.DataFrame([
        ("Max wells",               str(limits["max_wells"])),
        ("ACEI™ Score",             "✅" if limits["acei_enabled"]        else "🔒 Advanced+"),
        ("Scenario stress testing", "✅" if limits["scenarios_enabled"]   else "🔒 Advanced+"),
        ("Monte Carlo chart",       "✅" if limits["monte_carlo_visible"] else "🔒 Advanced+"),
        ("Portfolio ACEI™",         "✅" if limits["portfolio_enabled"]   else "🔒 Enterprise"),
        ("Excel export",            "✅" if limits["excel_export"]        else "🔒 Enterprise"),
        ("PDF report",              "✅"),
    ], columns=["Feature", "Your Plan"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<div style='font-size:0.82rem; color:#475569; margin-top:16px;'>"
        "To upgrade or renew, contact "
        "<a href='mailto:info@aquarisk.io' style='color:#3b82f6;'>info@aquarisk.io</a></div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

def render_dashboard(user, selected_scenario, threshold):
    st.title("Dashboard")

    wells = get_user_wells(user.user_id)
    if not wells:
        st.warning("No wells found. Go to **My Wells** to add a well first.")
        return

    # Build wells with data
    well_data = {}
    for w in wells:
        readings = get_well_readings(w["id"], user.user_id)
        if len(readings) >= 3:
            df_r = pd.DataFrame(readings)
            df_r["reading_date"] = pd.to_datetime(df_r["reading_date"])
            df_r = df_r.sort_values("reading_date")
            well_data[w["well_name"]] = {
                "id":     w["id"],
                "levels": df_r["water_level"].astype(float).values,
            }

    if not well_data:
        st.warning("No wells have enough data yet. Go to **Data Entry** and add at least 3 readings.")
        return

    selected_well_name = st.selectbox("Select Well", list(well_data.keys()))
    info       = well_data[selected_well_name]
    historical = info["levels"]
    well_id    = info["id"]
    last_value = float(historical[-1])

    x = np.arange(len(historical))
    slope, _ = np.polyfit(x, historical, 1)

    rain_coeff = RAIN_COEFF.get(selected_well_name, 0.012)
    pump_coeff = PUMP_COEFF.get(selected_well_name, 0.020)

    # ── Real weather data (Open-Meteo) ──
    # Try to get real precipitation; fall back to synthetic if unavailable
    real_rain, used_real_weather = get_real_rain_series(
        forecast_months=FORECAST_HORIZON_MONTHS
    )
    weather_summary = get_weather_summary()

    # Models — use real precipitation when available, else synthetic
    rain, pump, et = generate_scenario(FORECAST_HORIZON_MONTHS, selected_scenario)
    if used_real_weather:
        rain = apply_scenario_modifier(real_rain, selected_scenario)
    trend_f  = trend_model(last_value, slope, FORECAST_HORIZON_MONTHS)
    hybrid_f = hybrid_model(last_value, slope, FORECAST_HORIZON_MONTHS,
                            rain, pump, et, rain_coeff, pump_coeff)
    smooth_f = smoothed_model(last_value, slope, FORECAST_HORIZON_MONTHS)
    ensemble = weighted_ensemble(hybrid_f, trend_f, smooth_f)

    # Monte Carlo
    cleanup = patch_noise_sd(selected_well_name)
    sims, probability, p5, p95 = run_monte_carlo(ensemble, selected_well_name)
    cleanup()

    probability = float(np.asarray(probability).mean())
    risk        = classify_risk(probability)

    ensemble_plot = np.insert(ensemble, 0, last_value)
    p5_plot  = np.insert(p5,  0, last_value)
    p95_plot = np.insert(p95, 0, last_value)

    mean_cross, prob_12, prob_24 = calculate_time_to_threshold(sims, threshold)

    volatility = float(np.mean(p95 - p5))
    distance   = last_value - threshold
    acei_score, acei_category, acei_rec = calculate_acei(probability, slope*12, distance, volatility)

    # Log run
    log_analysis_run(user.user_id, well_id, selected_scenario, is_dry_run=False)

    st.markdown("---")

    # ── Weather Widget ──
    if weather_summary["available"]:
        di    = weather_summary["drought_index"]
        di_pct = int(di * 100)
        di_color = "#10b981" if di < 0.3 else "#f59e0b" if di < 0.65 else "#ef4444"
        di_label = "Normal" if di < 0.3 else "Moderate Drought" if di < 0.65 else "Severe Drought"
        lm   = weather_summary["last_month_mm"]
        fmm  = weather_summary["forecast_mm"]
        src  = weather_summary["source"]
        rain_tag = "🌧 Real precipitation data active" if used_real_weather else "⚠ Using synthetic precipitation"
        rain_tag_color = "#10b981" if used_real_weather else "#f59e0b"

        st.markdown(
            f"""<div style="background:#0e1e35; border:1px solid rgba(255,255,255,0.07);
                border-left:3px solid {di_color}; border-radius:10px;
                padding:14px 20px; margin-bottom:16px; display:flex;
                align-items:center; gap:24px; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.62rem; color:var(--muted, #4a5568);
                                text-transform:uppercase; letter-spacing:0.1em;
                                font-weight:700; margin-bottom:4px;">Precipitation Conditions</div>
                    <span style="color:{di_color}; font-weight:700; font-size:0.9rem;">{di_label}</span>
                    <span style="color:#4a5568; font-size:0.78rem; margin-left:10px;">
                        Last month: {lm} mm
                        {f"· Forecast: {fmm} mm/mo" if fmm else ""}
                        · Source: {src}
                    </span>
                </div>
                <div style="margin-left:auto; font-size:0.72rem;
                            color:{rain_tag_color}; font-weight:600;">
                    {rain_tag}
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    # Portfolio ACEI — Enterprise only
    if user.can("portfolio_enabled") and len(well_data) > 1:
        scores = []
        for wn, wd in well_data.items():
            h = wd["levels"]; lv = float(h[-1])
            xs = np.arange(len(h)); sl, _ = np.polyfit(xs, h, 1)
            rp, pp2, ee = generate_scenario(FORECAST_HORIZON_MONTHS, "baseline")
            rc = RAIN_COEFF.get(wn, 0.012); pc = PUMP_COEFF.get(wn, 0.020)
            tf = trend_model(lv, sl, FORECAST_HORIZON_MONTHS)
            hf = hybrid_model(lv, sl, FORECAST_HORIZON_MONTHS, rp, pp2, ee, rc, pc)
            sf = smoothed_model(lv, sl, FORECAST_HORIZON_MONTHS)
            ef = weighted_ensemble(hf, tf, sf)
            cl = patch_noise_sd(wn)
            _, pb, p5w, p95w = run_monte_carlo(ef, wn)
            cl()
            pb = float(np.asarray(pb).mean())
            sc, _, _ = calculate_acei(pb, sl*12, lv-threshold, float(np.mean(p95w-p5w)))
            scores.append(sc)

        ps = float(np.mean(scores))
        pc = "#10b981" if ps < 40 else "#f59e0b" if ps < 70 else "#ef4444"
        pl = "Low Exposure" if ps < 40 else "Moderate Exposure" if ps < 70 else "High Exposure"
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#0f1c30,#0d1a2e);
                border:1px solid {pc}44; border-left:4px solid {pc}; border-radius:12px;
                padding:24px 28px; margin-bottom:24px;">
                <div style="color:#64748b; font-size:0.72rem; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:6px;">Portfolio ACEI™</div>
                <div style="color:{pc}; font-size:2.4rem; font-weight:800; line-height:1;">
                    {ps:.1f}<span style="font-size:1rem; color:#64748b;"> / 100</span>
                </div>
                <div style="color:#e2e8f0; font-weight:600; margin-top:4px;">{pl}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Level",          f"{last_value:.2f} m")
    m2.metric("Exceedance Probability", f"{probability:.1%}")
    m3.metric("Risk Classification",    risk)
    m4.metric("Threshold Crossing",
              f"{mean_cross:.1f} mo" if mean_cross is not None else "Not reached")

    # ACEI
    if user.can("acei_enabled"):
        ac = "#10b981" if acei_score < 40 else "#f59e0b" if acei_score < 70 else "#ef4444"
        st.markdown(
            f"""<div style="background:#0f1c30; border-left:4px solid {ac}; border-radius:8px;
                padding:14px 20px; margin:12px 0; color:#cbd5e1; font-size:0.9rem;">
                <span style="color:{ac}; font-weight:700;">
                    ACEI™ {acei_score:.1f} / 100 — {acei_category}
                </span>
                &nbsp;·&nbsp; {acei_rec}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        locked_feature("ACEI™ Risk Score")

    if mean_cross is not None:
        tc1, tc2 = st.columns(2)
        tc1.metric("P(breach ≤ 12 months)", f"{prob_12:.1%}")
        tc2.metric("P(breach ≤ 24 months)", f"{prob_24:.1%}")

    st.markdown("---")

    # Forecast plot
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#080e1a"); ax.set_facecolor("#0d1526")
    x_hist = np.arange(len(historical))
    x_fore = np.arange(len(historical)-1, len(historical)-1+len(ensemble_plot))
    ax.plot(x_hist, historical, color="#60a5fa", linewidth=2.5, label="Historical", zorder=3)
    ax.plot(x_fore, ensemble_plot, color="#f59e0b", linewidth=2, linestyle="--",
            label="Forecast (Mean)", zorder=3)
    ax.fill_between(x_fore, p5_plot, p95_plot, color="#f59e0b", alpha=0.12, label="P5–P95")
    ax.axhline(threshold, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"Threshold ({threshold} m)")
    ax.axvline(len(historical)-1, color="#475569", linewidth=1, linestyle=":")
    ax.set_title(f"Well {selected_well_name}  ·  {selected_scenario.title()} Scenario",
                 color="#e2e8f0", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Months", color="#94a3b8"); ax.set_ylabel("Groundwater Level (m)", color="#94a3b8")
    ax.tick_params(colors="#64748b"); ax.spines[:].set_color("#1e293b")
    ax.grid(True, linestyle="--", alpha=0.15, color="#334155")
    ax.legend(facecolor="#0d1526", edgecolor="#1e293b", labelcolor="#cbd5e1", fontsize=9)
    st.pyplot(fig)

    os.makedirs("reports", exist_ok=True)
    chart_path = f"reports/forecast_{selected_well_name}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Monte Carlo chart
    if user.can("monte_carlo_visible"):
        st.markdown("#### Monte Carlo — Final Level Distribution")
        fig_mc, ax_mc = plt.subplots(figsize=(9, 3.5))
        fig_mc.patch.set_facecolor("#080e1a"); ax_mc.set_facecolor("#0d1526")
        fv = sims[:, -1]
        ax_mc.hist(fv, bins=40, color="#2563eb", alpha=0.75, edgecolor="none")
        ax_mc.axvline(threshold, color="#ef4444", linewidth=2, linestyle="--",
                      label=f"Threshold ({threshold} m)")
        ax_mc.axvline(float(np.mean(fv)), color="#f59e0b", linewidth=1.5, linestyle="--",
                      label=f"Mean ({np.mean(fv):.1f} m)")
        ax_mc.set_title("Distribution of Simulated Final Water Levels",
                        color="#e2e8f0", fontsize=11, pad=10)
        ax_mc.set_xlabel("Water Level (m)", color="#94a3b8")
        ax_mc.set_ylabel("Frequency", color="#94a3b8")
        ax_mc.tick_params(colors="#64748b"); ax_mc.spines[:].set_color("#1e293b")
        ax_mc.grid(True, linestyle="--", alpha=0.15, color="#334155")
        ax_mc.legend(facecolor="#0d1526", edgecolor="#1e293b", labelcolor="#cbd5e1", fontsize=9)
        st.pyplot(fig_mc); plt.close(fig_mc)
    else:
        locked_feature("Monte Carlo Risk Distribution")

    # Scenarios
    if user.can("scenarios_enabled"):
        st.markdown("---")
        st.markdown("#### Scenario Risk Comparison")
        sc_rows = []
        for sk in ["baseline", "drought", "expansion", "sustainable"]:
            rr, pp2, ee = generate_scenario(FORECAST_HORIZON_MONTHS, sk)
            tf2 = trend_model(last_value, slope, FORECAST_HORIZON_MONTHS)
            hf2 = hybrid_model(last_value, slope, FORECAST_HORIZON_MONTHS,
                               rr, pp2, ee, rain_coeff, pump_coeff)
            sf2 = smoothed_model(last_value, slope, FORECAST_HORIZON_MONTHS)
            ef2 = weighted_ensemble(hf2, tf2, sf2)
            cl2 = patch_noise_sd(selected_well_name)
            _, pb2, _, _ = run_monte_carlo(ef2, selected_well_name)
            cl2()
            pb2 = float(np.asarray(pb2).mean())
            sc_rows.append({"Scenario": sk.title(), "Prob. (%)": f"{pb2:.1%}",
                            "Risk": classify_risk(pb2)})
        st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)
    else:
        locked_feature("Scenario Stress Testing")

    # PDF
    st.markdown("---")
    st.markdown("#### Executive Report")
    if st.button("Generate Executive PDF"):
        pdf_bytes = generate_executive_pdf(
            well_name=selected_well_name,
            acei_score=acei_score if user.can("acei_enabled") else 0.0,
            acei_category=acei_category if user.can("acei_enabled") else "N/A",
            exceedance_probability=probability,
            risk_classification=risk,
            time_to_threshold=int(mean_cross) if mean_cross is not None else None,
            final_forecast_value=float(ensemble[-1]),
            threshold_value=threshold,
            chart_path=chart_path,
        )
        st.download_button(
            label="⬇ Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"AquaRisk_Report_{selected_well_name}.pdf",
            mime="application/pdf",
        )

    # Excel — Enterprise only
    if user.can("excel_export"):
        st.markdown("#### Data Export")
        export_df = pd.DataFrame([{
            "Well": selected_well_name, "Scenario": selected_scenario.title(),
            "Risk_Level": risk, "Exceedance_Probability": probability,
            "Mean_Crossing_Month": mean_cross, "P_Within_12M": prob_12, "P_Within_24M": prob_24,
        }])
        excel_path = f"reports/AquaRisk_{selected_well_name}.xlsx"
        export_df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button(
                label="⬇ Download Excel Export",
                data=f.read(),
                file_name=f"AquaRisk_{selected_well_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    user = st.session_state.user
    if user is None or not user.is_active:
        render_auth()
        return
    page, selected_scenario, threshold = render_sidebar(user)
    if page == "Dashboard":
        render_dashboard(user, selected_scenario, threshold)
    elif page == "My Wells":
        render_wells_page(user)
    elif page == "Data Entry":
        render_data_entry(user)
    elif page == "Account":
        render_account(user)


if __name__ == "__main__":
    main()