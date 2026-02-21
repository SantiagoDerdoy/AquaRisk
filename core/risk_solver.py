# core/risk_solver.py

import numpy as np
from core.scenario_engine import generate_scenario
from core.models import trend_model, hybrid_model, smoothed_model
from core.ensemble import weighted_ensemble
from core.monte_carlo import run_monte_carlo


def solve_pumping_reduction(
        last_value,
        well,
        forecast_horizon,
        rain_coeff,
        pump_coeff,
        target_risk=0.2):

    base_rain, base_pump, et = generate_scenario(
        forecast_horizon,
        scenario_type="baseline"
    )

    reduction_factors = np.linspace(0.0, 0.8, 15)

    for reduction in reduction_factors:

        adjusted_pump = base_pump * (1 - reduction)

        trend_forecast = trend_model(
            last_value,
            well,
            forecast_horizon
        )

        hybrid_forecast = hybrid_model(
            last_value,
            well,
            forecast_horizon,
            base_rain,
            adjusted_pump,
            et,
            rain_coeff,
            pump_coeff
        )

        smooth_forecast = smoothed_model(
            last_value,
            well,
            forecast_horizon
        )

        ensemble_forecast = weighted_ensemble(
            hybrid_forecast,
            trend_forecast,
            smooth_forecast
        )

        sims, probability, _, _ = run_monte_carlo(
            ensemble_forecast,
            well
        )

        if probability <= target_risk:
            return reduction, probability

    return None, None