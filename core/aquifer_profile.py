# core/aquifer_profile.py
# ─────────────────────────────────────────────
# AquaRisk — Aquifer Intelligence Engine
#
# Translates plain-language well profile answers
# into calibrated hydrogeological coefficients
# and a Model Confidence score (0–100%).
#
# Design philosophy:
#   - No technical jargon exposed to the client
#   - Every question uses observable, everyday language
#   - Internally maps to aquifer type + calibrated params
#   - Model Confidence rises as more answers are provided
# ─────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Defaults (Central Valley unconfined alluvial) ──────────
DEFAULT_RAIN_COEFF = 0.015
DEFAULT_PUMP_COEFF = 0.020
DEFAULT_NOISE_SD   = 0.30
DEFAULT_CONFIDENCE = 55  # baseline without any profile data


@dataclass
class AquiferCoefficients:
    rain_coeff:   float = DEFAULT_RAIN_COEFF
    pump_coeff:   float = DEFAULT_PUMP_COEFF
    noise_sd:     float = DEFAULT_NOISE_SD
    confidence:   int   = DEFAULT_CONFIDENCE   # 0-100
    aquifer_type: str   = "Unconfined alluvial (default)"
    notes:        list  = field(default_factory=list)


def calibrate_from_profile(profile: dict) -> AquiferCoefficients:
    """
    Given a plain-language well profile dict, return calibrated
    AquiferCoefficients and a model confidence score.

    Profile keys (all optional):
        artesian_pressure : bool   — water rises without pumping?
        rain_response     : str    — "fast" | "slow" | "none"
        well_depth_ft     : float  — depth in feet
        pump_rate_gpm     : float  — gallons per minute
        use_type          : str    — "irrigation" | "domestic" | "industrial"
        level_trend       : str    — "declining" | "stable" | "recovering"
    """
    if not profile:
        return AquiferCoefficients()

    rain_coeff   = DEFAULT_RAIN_COEFF
    pump_coeff   = DEFAULT_PUMP_COEFF
    noise_sd     = DEFAULT_NOISE_SD
    confidence   = DEFAULT_CONFIDENCE
    aquifer_type = "Unconfined alluvial (default)"
    notes        = []

    answered = sum(1 for k in [
        "artesian_pressure", "rain_response",
        "well_depth_ft", "pump_rate_gpm",
        "use_type", "level_trend"
    ] if k in profile and profile[k] not in (None, "", "unknown"))

    # ── 1. Artesian pressure → confined aquifer ──────────
    artesian = profile.get("artesian_pressure", False)
    if artesian:
        # Confined aquifer: less sensitive to rain, more stable
        rain_coeff   = 0.004
        noise_sd     = 0.18
        aquifer_type = "Confined artesian"
        notes.append("Confined aquifer detected — lower recharge sensitivity, more stable behavior.")
        confidence  += 12
    else:
        # Unconfined: more responsive to rain
        rain_coeff   = 0.018
        aquifer_type = "Unconfined alluvial"
        confidence  += 4

    # ── 2. Rain response → recharge rate ─────────────────
    rain_response = profile.get("rain_response", "")
    if rain_response == "fast":
        rain_coeff  *= 1.6   # very permeable — gravel/sand
        noise_sd    *= 0.85
        aquifer_type = aquifer_type + " (high permeability)"
        notes.append("Fast rain response — high permeability aquifer, stronger recharge signal.")
        confidence  += 10
    elif rain_response == "slow":
        rain_coeff  *= 0.7   # lower permeability — silt/clay
        noise_sd    *= 1.1
        aquifer_type = aquifer_type + " (low permeability)"
        notes.append("Slow rain response — lower permeability. Recharge lag modeled.")
        confidence  += 8
    elif rain_response == "none":
        rain_coeff  *= 0.3   # likely confined or very deep
        noise_sd    *= 0.9
        notes.append("No rain response — aquifer likely isolated or very deep.")
        confidence  += 6

    # ── 3. Well depth → extraction stress sensitivity ────
    depth_ft = profile.get("well_depth_ft")
    if depth_ft:
        depth_ft = float(depth_ft)
        if depth_ft < 100:
            pump_coeff  *= 1.4   # shallow — more sensitive to extraction
            noise_sd    *= 1.15
            notes.append("Shallow well (<100 ft) — higher sensitivity to pumping stress.")
            confidence  += 8
        elif depth_ft < 300:
            pump_coeff  *= 1.0   # typical
            confidence  += 7
        elif depth_ft < 600:
            pump_coeff  *= 0.75  # deeper — more buffered
            noise_sd    *= 0.9
            notes.append("Deep well (300–600 ft) — buffered extraction response.")
            confidence  += 7
        else:
            pump_coeff  *= 0.55  # very deep — likely confined
            noise_sd    *= 0.8
            notes.append("Very deep well (>600 ft) — confined or semi-confined behavior assumed.")
            confidence  += 8

    # ── 4. Pump rate → extraction scale ──────────────────
    pump_rate_gpm = profile.get("pump_rate_gpm")
    if pump_rate_gpm:
        pump_rate_gpm = float(pump_rate_gpm)
        # Scale pump_coeff relative to typical agricultural rate (300-600 gpm)
        typical = 450.0
        scale   = pump_rate_gpm / typical
        pump_coeff = pump_coeff * (0.6 + 0.4 * scale)  # partial scaling
        notes.append(f"Pump rate {pump_rate_gpm:.0f} gpm calibrated into extraction model.")
        confidence += 9

    # ── 5. Use type → pumping pattern ────────────────────
    use_type = profile.get("use_type", "")
    if use_type == "irrigation":
        # Seasonal pumping peaks — higher uncertainty in summer
        noise_sd   *= 1.1
        confidence += 5
    elif use_type == "domestic":
        # Low, steady extraction
        pump_coeff *= 0.35
        noise_sd   *= 0.85
        notes.append("Domestic use — low extraction rate, high model stability.")
        confidence += 5
    elif use_type == "industrial":
        # High, year-round extraction
        pump_coeff *= 1.3
        notes.append("Industrial use — elevated year-round extraction modeled.")
        confidence += 5

    # ── 6. Observed trend → slope correction hint ────────
    level_trend = profile.get("level_trend", "")
    if level_trend == "declining":
        notes.append("Observed declining trend confirmed — model weights declining scenarios higher.")
        confidence += 4
    elif level_trend == "recovering":
        notes.append("Recovering trend noted — model accounts for recharge cycle.")
        confidence += 4
    elif level_trend == "stable":
        noise_sd   *= 0.9
        confidence += 3

    # Cap confidence at 96 — never claim 100% certainty
    confidence = min(confidence, 96)

    # Round coefficients to 4 decimal places
    rain_coeff = round(rain_coeff, 4)
    pump_coeff = round(pump_coeff, 4)
    noise_sd   = round(noise_sd,   4)

    return AquiferCoefficients(
        rain_coeff   = rain_coeff,
        pump_coeff   = pump_coeff,
        noise_sd     = noise_sd,
        confidence   = confidence,
        aquifer_type = aquifer_type,
        notes        = notes,
    )


def confidence_label(score: int) -> tuple[str, str]:
    """Returns (label, color_hex) for a confidence score."""
    if score >= 85:
        return "High Confidence", "#10b981"
    elif score >= 70:
        return "Good Confidence", "#60a5fa"
    elif score >= 55:
        return "Baseline", "#f59e0b"
    else:
        return "Low — Add Profile Data", "#ef4444"


def save_well_profile(
    well_id: str,
    user_id: str,
    profile: dict,
) -> Optional[str]:
    """
    Save aquifer profile to Supabase.
    Returns error string or None on success.
    """
    import json
    from datetime import date
    try:
        from core.supabase_client import get_supabase
        profile_with_date = {**profile, "completed_at": str(date.today())}
        get_supabase().table("wells") \
            .update({"aquifer_profile": profile_with_date}) \
            .eq("id", well_id) \
            .eq("user_id", user_id) \
            .execute()
        return None
    except Exception as e:
        return str(e)


def get_well_coefficients(well: dict) -> AquiferCoefficients:
    """
    Given a well dict (from Supabase), return calibrated coefficients.
    Falls back to config.py defaults if no profile saved.
    """
    profile = well.get("aquifer_profile") or {}
    if isinstance(profile, str):
        import json
        try:
            profile = json.loads(profile)
        except Exception:
            profile = {}
    return calibrate_from_profile(profile)
