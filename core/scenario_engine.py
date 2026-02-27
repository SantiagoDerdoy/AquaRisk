# core/scenario_engine.py
# ─────────────────────────────────────────────
# AquaRisk — Scenario Engine
#
# CONVENTION: water_level = hydraulic head in meters above sea level
#   - Higher value = more water available = GOOD
#   - Lower value  = less water available = RISK
#   - Under drought: levels DROP (go lower)
#   - Under recharge/sustainable: levels RISE (go higher)
#
# The hybrid_model formula is:
#   next_level = current + slope + rain_coeff*rain - pump_coeff*pump - et_coeff*et
#
# So:
#   - More rain  → levels RISE  (positive effect)
#   - More pump  → levels FALL  (negative effect)
#   - More ET    → levels FALL  (negative effect)
#
# Scenario design: drought must produce NEGATIVE net climate_effect
# so that levels decline meaningfully relative to baseline.
# ─────────────────────────────────────────────

import numpy as np


# ── Reference coefficients (Central Valley defaults) ──
# These represent how 1 unit of each driver affects the level (m/month)
# Override per-well in config.py via RAIN_COEFF and PUMP_COEFF

DEFAULT_RAIN_COEFF = 0.015   # m of level rise per mm of rain
DEFAULT_PUMP_COEFF = 0.020   # m of level drop per M m3 pumped
DEFAULT_ET_COEFF   = 0.008   # m of level drop per mm of ET


def generate_scenario(months: int, scenario_type: str) -> tuple:
    """
    Generates monthly climate & pumping driver arrays for a given scenario.

    Units:
        rain : mm/month  — precipitation
        pump : Mm³/month — extraction (million cubic meters)
        et   : mm/month  — evapotranspiration

    Scenario logic (hydraulic head convention):
        baseline    → moderate rain, normal pumping → slight decline (typical managed aquifer)
        drought     → low rain, HIGH pumping, high ET → significant decline
        expansion   → moderate rain, VERY HIGH pumping → accelerated decline
        sustainable → good rain, LOW pumping, low ET → stable or slight recovery

    Parameters
    ----------
    months        : int — forecast horizon
    scenario_type : str — one of baseline | drought | expansion | sustainable

    Returns
    -------
    (rain, pump, et) as numpy arrays of length `months`
    """
    if scenario_type == "baseline":
        # Normal operations: balanced recharge and extraction
        # Net climate effect ≈ slightly negative (aquifer in slow managed decline)
        rain = np.full(months, 48.0)   # mm/month
        pump = np.full(months, 16.0)   # Mm³/month — calibrated for slight net decline
        et   = np.full(months, 30.0)   # mm/month

    elif scenario_type == "drought":
        # Severe drought: 60% less rain, farmers pump heavily to compensate
        # Net effect: significant level decline each month
        rain = np.full(months, 18.0)   # 62% reduction vs baseline
        pump = np.full(months, 28.0)   # massive increase — farmers compensate
        et   = np.full(months, 48.0)   # high ET from heat stress

    elif scenario_type == "expansion":
        # Production expansion: new irrigated acreage, heavy extraction
        rain = np.full(months, 44.0)   # near normal
        pump = np.full(months, 34.0)   # very high extraction
        et   = np.full(months, 36.0)   # moderately elevated

    elif scenario_type == "sustainable":
        # Conservation: managed recharge program, reduced extraction
        # Net effect: stable or slight recovery
        rain = np.full(months, 58.0)   # slightly above normal
        pump = np.full(months, 10.0)   # significant reduction
        et   = np.full(months, 24.0)   # lower ET

    else:
        raise ValueError(
            f"Unknown scenario: '{scenario_type}'. "
            "Valid options: baseline, drought, expansion, sustainable."
        )

    return rain, pump, et


def get_scenario_net_effect(
    scenario_type: str,
    rain_coeff: float = DEFAULT_RAIN_COEFF,
    pump_coeff: float = DEFAULT_PUMP_COEFF,
    et_coeff:   float = DEFAULT_ET_COEFF,
) -> float:
    """
    Returns the expected monthly climate effect (m/month) for a scenario,
    excluding the historical trend slope.

    Positive = level rising, Negative = level falling.
    Useful for validating that scenarios make physical sense.
    """
    rain, pump, et = generate_scenario(12, scenario_type)
    effect = (
        rain_coeff * float(rain[0])
        - pump_coeff * float(pump[0])
        - et_coeff   * float(et[0])
    )
    return round(effect, 4)


def validate_scenarios(
    rain_coeff: float = DEFAULT_RAIN_COEFF,
    pump_coeff: float = DEFAULT_PUMP_COEFF,
) -> dict:
    """
    Returns a summary of net monthly climate effects for all scenarios.
    Use this to verify scenarios are physically consistent:
        drought    should be the most negative
        expansion  should be negative and larger than baseline
        baseline   should be slightly negative
        sustainable should be least negative or positive
    """
    results = {}
    for s in ["baseline", "drought", "expansion", "sustainable"]:
        effect = get_scenario_net_effect(s, rain_coeff, pump_coeff)
        results[s] = {
            "net_monthly_effect_m": effect,
            "direction": "↑ Rising" if effect > 0 else "↓ Falling",
            "physically_correct": (
                (s == "drought"     and effect < -0.05) or
                (s == "expansion"   and effect < -0.05) or
                (s == "baseline"    and effect < 0) or
                (s == "sustainable" and effect > -0.02)
            )
        }
    return results