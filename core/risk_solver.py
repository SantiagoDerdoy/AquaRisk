# core/risk_solver.py
import numpy as np
from core.scenario_engine import generate_scenario
from core.models import trend_model, hybrid_model, smoothed_model
from core.ensemble import weighted_ensemble
from core.monte_carlo import run_monte_carlo
from core.feature_engineering import calculate_rate_of_decline


def solve_pumping_reduction(
        last_value,
        well,
        forecast_horizon,
        rain_coeff,
        pump_coeff,
        historical_levels,
        target_risk=0.2):
    """
    Iterates over pumping reduction factors to find the minimum reduction
    needed to bring exceedance probability below target_risk.

    Parameters
    ----------
    last_value        : float  – last observed groundwater level
    well              : str    – well ID (used for Monte Carlo noise lookup)
    forecast_horizon  : int    – number of months to forecast
    rain_coeff        : dict   – {well_id: coefficient}
    pump_coeff        : dict   – {well_id: coefficient}
    historical_levels : array  – historical groundwater levels for slope calc
    target_risk       : float  – exceedance probability target (default 0.20)
    """

    slope = calculate_rate_of_decline(historical_levels)

    base_rain, base_pump, et = generate_scenario(
        forecast_horizon,
        scenario_type="baseline"
    )

    reduction_factors = np.linspace(0.0, 0.8, 15)

    for reduction in reduction_factors:

        adjusted_pump = base_pump * (1 - reduction)

        trend_forecast = trend_model(
            last_value,
            slope,
            forecast_horizon
        )

        hybrid_forecast = hybrid_model(
            last_value,
            slope,
            forecast_horizon,
            base_rain,
            adjusted_pump,
            et,
            rain_coeff[well],
            pump_coeff[well]
        )

        smooth_forecast = smoothed_model(
            last_value,
            slope,
            forecast_horizon
        )

        ensemble_forecast = weighted_ensemble(
            hybrid_forecast,
            trend_forecast,
            smooth_forecast
        )

        _, probability, _, _ = run_monte_carlo(
            ensemble_forecast,
            well
        )

        probability = float(np.asarray(probability).mean())

        if probability <= target_risk:
            return reduction, probability

    return None, None