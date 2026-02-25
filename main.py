# main.py

import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from config import (
    CLIENT_NAME,
    LOCATION,
    FORECAST_HORIZON_MONTHS,
    THRESHOLD_LEVEL,
    RAIN_COEFF,
    PUMP_COEFF
)
from core.data_processing import load_dataset, validate_dataset
from core.feature_engineering import calculate_rate_of_decline
from core.scenario_engine import generate_scenario
from core.models import trend_model, hybrid_model, smoothed_model
from core.ensemble import weighted_ensemble
from core.monte_carlo import run_monte_carlo
from core.risk_engine import classify_risk, calculate_time_to_threshold
from core.risk_solver import solve_pumping_reduction
from core.report_generator import generate_report
from core.plot_generator import generate_forecast_plot
from core.acei_engine import calculate_acei
from datetime import datetime


def run_model():

    df = load_dataset("data/demo_dataset.csv")
    df = validate_dataset(df)

    df = df.rename(columns={
        "well": "Well_ID",
        "water_level": "Groundwater_Level_m"
    })

    wells = df["Well_ID"].unique()

    SCENARIOS = {
        "baseline":    "Baseline Operating Scenario",
        "drought":     "Climatic Stress Scenario",
        "expansion":   "Production Expansion Scenario",
        "sustainable": "Sustainability Strategy Scenario"
    }

    print("=" * 52)
    print("AquaRisk Predictive Groundwater Intelligence Demo")
    print(f"Client:           {CLIENT_NAME}")
    print(f"Location:         {LOCATION}")
    print(f"Forecast Horizon: {FORECAST_HORIZON_MONTHS} months")
    print("=" * 52, "\n")

    report_results = []

    for well in wells:

        well_df = df[df["Well_ID"] == well]
        historical_levels = well_df["Groundwater_Level_m"].values
        last_value = float(historical_levels[-1])

        slope = calculate_rate_of_decline(historical_levels)

        print("=" * 52)
        print(f"Well: {well}")
        print(f"Current Level:        {last_value:.2f} m")
        print(f"Historical Decline:   {abs(slope) * 12:.2f} m/year")
        print("-" * 52)

        scenario_results = []

        # ── Run all scenarios ──────────────────────────────────────────────
        for scenario_key, scenario_name in SCENARIOS.items():

            rain, pump, et = generate_scenario(
                FORECAST_HORIZON_MONTHS,
                scenario_type=scenario_key
            )

            # All three models receive (last_value, slope, months, ...)
            trend_forecast = trend_model(last_value, slope, FORECAST_HORIZON_MONTHS)

            hybrid_forecast = hybrid_model(
                last_value,
                slope,
                FORECAST_HORIZON_MONTHS,
                rain,
                pump,
                et,
                RAIN_COEFF[well],
                PUMP_COEFF[well]
            )

            smooth_forecast = smoothed_model(
                last_value,
                slope,
                FORECAST_HORIZON_MONTHS
            )

            ensemble_forecast = weighted_ensemble(
                hybrid_forecast,
                trend_forecast,
                smooth_forecast
            )

            sims, probability, p5, p95 = run_monte_carlo(ensemble_forecast, well)
            probability = float(np.asarray(probability).mean())

            risk = classify_risk(probability)
            scenario_results.append((scenario_name, probability, risk))

        # Keep last scenario's outputs for per-well reporting
        volatility = float(np.mean(p95 - p5))
        distance_to_threshold = last_value - THRESHOLD_LEVEL

        acei_score, acei_category, acei_recommendation = calculate_acei(
            probability,
            slope * 12,
            distance_to_threshold,
            volatility
        )

        print(f"\nACEI™ Score : {acei_score} / 100")
        print(f"Category    : {acei_category}")
        print(f"Advisory    : {acei_recommendation}")

        mean_cross, prob_12, prob_24 = calculate_time_to_threshold(
            sims, THRESHOLD_LEVEL
        )

        # ── Plot ───────────────────────────────────────────────────────────
        plot_filename = os.path.join(RESULTS_DIR, f"{well}_forecast.png")
        generate_forecast_plot(
            historical_levels,
            ensemble_forecast,
            p5,
            p95,
            THRESHOLD_LEVEL,
            plot_filename
        )

        # ── Scenario ranking ───────────────────────────────────────────────
        scenario_results.sort(key=lambda x: x[1], reverse=True)

        print("\nScenario Risk Ranking:")
        for s, p, r in scenario_results:
            print(f"  {s:<40} | {p:.2%} | {r}")

        highest = scenario_results[0]
        lowest  = scenario_results[-1]

        print("\nExecutive Insight:")
        if mean_cross is not None:
            print(f"  Expected threshold crossing : {mean_cross:.1f} months")
            print(f"  P(breach within 12 months) : {prob_12:.2%}")
            print(f"  P(breach within 24 months) : {prob_24:.2%}")
        else:
            print("  Threshold not expected to be crossed within forecast horizon.")

        print(
            f"  Highest risk: {highest[0]} — {highest[1]:.2%} exceedance probability."
        )
        print(
            f"  Lowest risk:  {lowest[0]} — {lowest[1]:.2%} probability."
        )

        # ── Pumping optimisation ───────────────────────────────────────────
        reduction, achieved_risk = solve_pumping_reduction(
            last_value,
            well,
            FORECAST_HORIZON_MONTHS,
            RAIN_COEFF,
            PUMP_COEFF,
            historical_levels=historical_levels,
            target_risk=0.2
        )

        if reduction is not None:
            print(
                f"\n  Optimisation: pumping must be reduced by ~{reduction*100:.0f}% "
                f"to achieve {achieved_risk:.2%} risk."
            )
        else:
            print("\n  Even with 80% pumping reduction, risk remains above target.")

        # ── Collect for report ─────────────────────────────────────────────
        scenario_output = []
        for s, p, r in scenario_results:
            scenario_output.append({
                "name":       s.upper(),
                "probability": p,
                "risk":       r,
                "mean_cross": mean_cross,
                "prob_12":    prob_12,
                "prob_24":    prob_24
            })

        report_results.append({
            "well":         well,
            "scenarios":    scenario_output,
            "optimization": {"reduction": reduction, "risk": achieved_risk},
            "plot":         plot_filename
        })

        print("=" * 52, "\n")

    # ── Excel export ───────────────────────────────────────────────────────
    excel_rows = []
    for wd in report_results:
        for scenario in wd["scenarios"]:
            excel_rows.append({
                "Well":                     wd["well"],
                "Scenario":                 scenario["name"],
                "Risk_Level":               scenario["risk"],
                "Exceedance_Probability":   scenario["probability"],
                "Mean_Crossing_Month":      scenario["mean_cross"],
                "Probability_Within_12M":   scenario["prob_12"],
                "Probability_Within_24M":   scenario["prob_24"]
            })

    excel_df = pd.DataFrame(excel_rows)
    excel_path = os.path.join(RESULTS_DIR, "AquaRisk_Data.xlsx")
    excel_df.to_excel(excel_path, index=False)
    print(f"Excel exported → {excel_path}")

    # ── PDF report ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(RESULTS_DIR, f"AquaRisk_Report_{timestamp}.pdf")

    generate_report(pdf_path, CLIENT_NAME, LOCATION, report_results)
    print(f"PDF report generated → {pdf_path}")


if __name__ == "__main__":
    run_model()