# main.py

import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
import pandas as pd
from config import *
from core.data_processing import load_dataset, validate_dataset
from core.feature_engineering import calculate_rate_of_decline
from core.scenario_engine import generate_scenario
from core.models import trend_model, hybrid_model, smoothed_model
from core.ensemble import weighted_ensemble
from core.monte_carlo import run_monte_carlo
from core.risk_engine import classify_risk
from core.risk_solver import solve_pumping_reduction
from core.report_generator import generate_report
from core.plot_generator import generate_forecast_plot
from core.risk_engine import classify_risk, calculate_time_to_threshold
from datetime import datetime

# Coeficientes hidrogeológicos (demo)
RAIN_COEFF = {
    "GW-01": 0.015,
    "GW-02": 0.010,
    "GW-03": 0.008,
    "GW-04": 0.020
}

PUMP_COEFF = {
    "GW-01": 0.020,
    "GW-02": 0.015,
    "GW-03": 0.030,
    "GW-04": 0.018
}


def run_model():

    df = load_dataset("data/demo_dataset.csv")
    df = validate_dataset(df)

    # 🔥 NORMALIZAR NOMBRES DE COLUMNAS
    df = df.rename(columns={
        "well": "Well_ID",
        "water_level": "Groundwater_Level_m"
    })

    print(df.columns)  # Debug opcional

    wells = df["Well_ID"].unique()

    SCENARIOS = [
        "baseline",
        "drought",
        "expansion",
        "sustainable"
    ]

    print("====================================================")
    print("AquaRisk Predictive Groundwater Intelligence Demo")
    print(f"Client: {CLIENT_NAME}")
    print(f"Location: {LOCATION}")
    print("Forecast Horizon:", FORECAST_HORIZON_MONTHS, "months")
    print("====================================================\n")

    report_results = []

    for well in wells:

        well_df = df[df["Well_ID"] == well]
        historical_levels = well_df["Groundwater_Level_m"].values
        last_value = well_df["Groundwater_Level_m"].iloc[-1]

        slope = calculate_rate_of_decline(
            well_df["Groundwater_Level_m"].values
        )

        print("====================================================")
        print(f"Well: {well}")
        print(f"Current Level: {last_value:.2f} m")
        print(f"Historical Decline Rate: {abs(slope)*12:.2f} m/year")
        print("----------------------------------------------------")

        scenario_results = []

        for scenario in SCENARIOS:

            rain, pump, et = generate_scenario(
                FORECAST_HORIZON_MONTHS,
                scenario_type=scenario
            )

            trend_forecast = trend_model(
                last_value,
                well,
                FORECAST_HORIZON_MONTHS
            )

            hybrid_forecast = hybrid_model(
                last_value,
                well,
                FORECAST_HORIZON_MONTHS,
                rain,
                pump,
                et,
                RAIN_COEFF,
                PUMP_COEFF
            )

            smooth_forecast = smoothed_model(
                last_value,
                well,
                FORECAST_HORIZON_MONTHS
            )

            ensemble_forecast = weighted_ensemble(
                hybrid_forecast,
                trend_forecast,
                smooth_forecast
            )

            sims, probability, p5, p95 = run_monte_carlo(
                ensemble_forecast,
                well
            )

            mean_cross, prob_12, prob_24 = calculate_time_to_threshold(
                sims,
                THRESHOLD_LEVEL
            )

            plot_filename = os.path.join(RESULTS_DIR, f"{well}_forecast.png")

            generate_forecast_plot(
                historical_levels,
                ensemble_forecast,
                p5,
                p95,
                THRESHOLD_LEVEL,
                plot_filename
            )
          
            risk = classify_risk(probability)

            scenario_results.append(
                (scenario, probability, risk)
            )

        # Ordenar por riesgo
        scenario_results.sort(key=lambda x: x[1], reverse=True)

        print("Scenario Risk Ranking:")
        for s, p, r in scenario_results:
            print(f"{s.upper():<15} | {p:.2%} | {r}")

        highest = scenario_results[0]
        lowest = scenario_results[-1]

        print("\nExecutive Insight:")
        
        if mean_cross is not None:
            print(f"Expected threshold crossing: {mean_cross:.1f} months")
            print(f"Probability within 12 months: {prob_12:.2%}")
            print(f"Probability within 24 months: {prob_24:.2%}")
        else:
            print("Threshold not expected to be crossed within forecast horizon.")
        
        print(
            f"Highest risk observed under {highest[0].upper()} scenario "
            f"with {highest[1]:.2%} exceedance probability."
        )

        print(
            f"Lowest risk observed under {lowest[0].upper()} scenario "
            f"with {lowest[1]:.2%} probability."
        )

        reduction, achieved_risk = solve_pumping_reduction(
            last_value,
            well,
            FORECAST_HORIZON_MONTHS,
            RAIN_COEFF,
            PUMP_COEFF,
            target_risk=0.2
        )

        if reduction is not None:
            print(
                f"To reduce risk below 20%, pumping must be reduced "
                f"by approximately {reduction*100:.0f}%."
            )
            print(f"Achieved Risk: {achieved_risk:.2%}")
        else:
            print(
                "Even with 80% pumping reduction, risk remains above target."
            )

        # Preparar datos para reporte
        scenario_output = []
        for s, p, r in scenario_results:
            scenario_output.append({
                "name": s.upper(),
                "probability": p,
                "risk": r,
                "mean_cross": mean_cross,
                "prob_12": prob_12,
                "prob_24": prob_24
            })

        optimization_data = {
            "reduction": reduction,
            "risk": achieved_risk
        }

        report_results.append({
            "well": well,
            "scenarios": scenario_output,
            "optimization": optimization_data,
            "plot": plot_filename
        })

        print("====================================================\n")

    excel_rows = []

    for well_data in report_results:
        for scenario in well_data["scenarios"]:
            excel_rows.append({
                "Well": well_data["well"],
                "Scenario": scenario["name"],
                "Risk_Level": scenario["risk"],
                "Exceedance_Probability": scenario["probability"],
                "Mean_Crossing_Month": scenario["mean_cross"],
                "Probability_Within_12M": scenario["prob_12"],
                "Probability_Within_24M": scenario["prob_24"]
            })

    excel_df = pd.DataFrame(excel_rows)

    excel_path = os.path.join(RESULTS_DIR, "AquaRisk_Data.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print("Excel Export Generated: AquaRisk_Data.xlsx")

    # Generar PDF final
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(
        RESULTS_DIR,
        f"AquaRisk_Report_{timestamp}.pdf"
    )

    generate_report(
        pdf_path,
        CLIENT_NAME,
        LOCATION,
        report_results
    )

    print("PDF Report Generated: AquaRisk_Report.pdf")

if __name__ == "__main__":
    run_model()