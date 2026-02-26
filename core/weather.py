# core/weather.py
# ─────────────────────────────────────────────
# AquaRisk — Real Weather Data via Open-Meteo
# Free API, no key required.
# Fetches historical precipitation + forecast
# for a given lat/lon and integrates with the
# scenario engine to replace synthetic rain data.
# ─────────────────────────────────────────────

from __future__ import annotations
import urllib.request
import json
from datetime import date, timedelta
from typing import Optional
import numpy as np


# ── Default coordinates ───────────────────────
# California Central Valley (Fresno area) as default
# Users can override per-well via the DB
DEFAULT_LAT  = 36.7378
DEFAULT_LON  = -119.7871


# ── Public API endpoint ───────────────────────
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"


def fetch_historical_precipitation(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    months_back: int = 24,
) -> Optional[list[float]]:
    """
    Fetch monthly precipitation totals (mm) for the past N months
    from Open-Meteo Historical Weather API.

    Returns a list of monthly totals ordered oldest → newest,
    or None if the request fails.
    """
    end_date   = date.today().replace(day=1) - timedelta(days=1)
    start_date = (end_date - timedelta(days=months_back * 31)).replace(day=1)

    url = (
        f"{OPEN_METEO_BASE}/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=precipitation_sum"
        f"&timezone=America%2FLos_Angeles"
    )

    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())

        dates  = data["daily"]["time"]
        precip = data["daily"]["precipitation_sum"]

        # Aggregate daily → monthly
        monthly: dict[str, float] = {}
        for d, p in zip(dates, precip):
            month_key = d[:7]  # "YYYY-MM"
            monthly[month_key] = monthly.get(month_key, 0.0) + (p or 0.0)

        # Return sorted by month
        return [monthly[k] for k in sorted(monthly.keys())]

    except Exception:
        return None


def fetch_forecast_precipitation(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    days: int = 90,
) -> Optional[float]:
    """
    Fetch the next N days of forecast precipitation and return
    the estimated monthly average (mm/month).

    Returns None on failure.
    """
    url = (
        f"{OPEN_METEO_BASE}/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum"
        f"&forecast_days={min(days, 16)}"  # API max is 16 days
        f"&timezone=America%2FLos_Angeles"
    )

    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())

        daily_totals = [p or 0.0 for p in data["daily"]["precipitation_sum"]]
        n_days = len(daily_totals)
        if n_days == 0:
            return None

        # Annualize to monthly equivalent
        daily_avg    = sum(daily_totals) / n_days
        monthly_avg  = daily_avg * 30.44
        return monthly_avg

    except Exception:
        return None


def get_real_rain_series(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    forecast_months: int = 24,
    historical_months: int = 24,
) -> tuple[np.ndarray, bool]:
    """
    Build a precipitation series of length `forecast_months` (mm/month)
    by combining:
      1. Historical monthly averages from Open-Meteo
      2. Forecast precipitation for near-term months
      3. Climatological fallback if API unavailable

    Returns:
        (rain_array, used_real_data)
        used_real_data = True if real API data was used
    """
    historical = fetch_historical_precipitation(lat, lon, historical_months)
    forecast   = fetch_forecast_precipitation(lat, lon)

    if historical and len(historical) >= 6:
        # Use last 12 months as seasonal baseline
        baseline   = np.array(historical[-12:]) if len(historical) >= 12 else np.array(historical)
        monthly_avg = float(np.mean(baseline))
        monthly_std = float(np.std(baseline)) if len(baseline) > 1 else monthly_avg * 0.3

        # Use forecast for first month if available
        first_month = forecast if forecast is not None else monthly_avg

        # Build forecast array with realistic seasonal variation
        np.random.seed(42)
        rain = np.array([
            max(0.0, first_month if i == 0 else np.random.normal(monthly_avg, monthly_std))
            for i in range(forecast_months)
        ])
        return rain, True

    # Fallback: Central Valley climatology (dry summers, wet winters)
    rain = _central_valley_climatology(forecast_months)
    return rain, False


def apply_scenario_modifier(
    rain_base: np.ndarray,
    scenario: str,
) -> np.ndarray:
    """
    Apply scenario-specific multipliers to a precipitation series.
    This allows real weather data to still reflect scenario stress.
    """
    modifiers = {
        "baseline":    1.00,
        "drought":     0.45,   # 55% reduction — severe drought
        "expansion":   0.90,   # slight reduction from land use change
        "sustainable": 1.15,   # conservation + managed recharge
    }
    multiplier = modifiers.get(scenario, 1.0)
    return rain_base * multiplier


def _central_valley_climatology(months: int) -> np.ndarray:
    """
    Fallback: Approximate monthly precipitation (mm) for
    California Central Valley using climatological averages.
    Mediterranean climate: wet Nov–Mar, dry Jun–Sep.
    """
    # Monthly averages (mm) for Fresno, CA
    MONTHLY_AVG = [
        44, 38, 35, 18, 12, 3, 1, 2, 7, 20, 30, 38
    ]  # Jan–Dec

    today  = date.today()
    result = []
    for i in range(months):
        month_idx = (today.month - 1 + i) % 12
        avg = MONTHLY_AVG[month_idx]
        # Add small noise
        val = max(0.0, np.random.normal(avg, avg * 0.25))
        result.append(val)

    return np.array(result)


def get_weather_summary(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """
    Returns a summary dict for display in the dashboard.
    {
        "available": bool,
        "last_month_mm": float,
        "forecast_mm": float,
        "drought_index": float,   # 0 (wet) to 1 (extreme drought)
        "source": str,
    }
    """
    historical = fetch_historical_precipitation(lat, lon, months_back=13)

    if not historical or len(historical) < 2:
        return {
            "available":     False,
            "last_month_mm": None,
            "forecast_mm":   None,
            "drought_index": None,
            "source":        "unavailable",
        }

    last_month     = historical[-1]
    twelve_mo_avg  = float(np.mean(historical[-12:])) if len(historical) >= 12 else float(np.mean(historical))
    forecast_mm    = fetch_forecast_precipitation(lat, lon)

    # Drought index: how much below average is this month (0 = normal, 1 = extreme)
    if twelve_mo_avg > 0:
        drought_index = max(0.0, min(1.0, 1.0 - last_month / twelve_mo_avg))
    else:
        drought_index = 0.0

    return {
        "available":     True,
        "last_month_mm": round(last_month, 1),
        "forecast_mm":   round(forecast_mm, 1) if forecast_mm else None,
        "drought_index": round(drought_index, 2),
        "source":        "Open-Meteo",
    }
