# core/models.py

import numpy as np


def trend_model(last_value, slope, months):
    """
    Linear trend model.
    slope is expressed in meters per month.
    """

    forecast = []
    level = float(last_value)

    for _ in range(months):
        level = level + slope
        forecast.append(level)

    forecast = np.array(forecast)

    if not np.all(np.isfinite(forecast)):
        raise ValueError("Trend model diverged. Check slope value.")

    return forecast


def hybrid_model(
    last_value,
    slope,
    months,
    rain,
    pump,
    et,
    rain_coeff,
    pump_coeff,
    et_coeff=0.0
):
    """
    Hybrid physical-statistical model.

    All climate drivers must be scaled consistently.
    Pumping MUST be expressed in million m3/month.
    """

    rain = np.array(rain, dtype=float)
    pump = np.array(pump, dtype=float)
    et = np.array(et, dtype=float)

    forecast = []
    current = float(last_value)

    for i in range(months):
        climate_effect = (
            rain_coeff * rain[i]
            - pump_coeff * pump[i]
            - et_coeff * et[i]
        )

        current = current + slope + climate_effect
        forecast.append(current)

    forecast = np.array(forecast)

    if not np.all(np.isfinite(forecast)):
        raise ValueError("Hybrid model diverged. Check coefficient scaling.")

    return forecast


def smoothed_model(last_value, slope, months, smoothing_factor=0.8):
    """
    Smoothed trend model.
    """

    forecast = []
    level = float(last_value)

    for _ in range(months):
        level = level + slope * smoothing_factor
        forecast.append(level)

    forecast = np.array(forecast)

    if not np.all(np.isfinite(forecast)):
        raise ValueError("Smoothed model diverged.")

    return forecast