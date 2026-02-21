# plot_generator.py

import numpy as np
import matplotlib.pyplot as plt


def generate_forecast_plot(
        historical_levels,
        ensemble_forecast,
        p5,
        p95,
        threshold,
        filename):

    history_len = len(historical_levels)
    forecast_len = len(ensemble_forecast)

    history_months = np.arange(history_len)
    forecast_months = np.arange(history_len, history_len + forecast_len)

    plt.figure(figsize=(10, 6))

    # Histórico real
    plt.plot(history_months, historical_levels,
             color="black",
             linewidth=2,
             label="Historical Observed")

    # Forecast medio
    plt.plot(forecast_months, ensemble_forecast,
             color="blue",
             linewidth=2,
             label="Forecast (Mean)")

    # Banda incertidumbre
    plt.fill_between(forecast_months, p5, p95,
                     color="blue",
                     alpha=0.2,
                     label="Uncertainty (P5–P95)")

    # Threshold crítico
    plt.axhline(y=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Critical Threshold")

    # Línea inicio forecast
    plt.axvline(x=history_len - 1,
                color="gray",
                linestyle=":",
                label="Forecast Start")

    plt.xlabel("Months")
    plt.ylabel("Groundwater Level (m)")
    plt.title("Groundwater Risk Projection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()