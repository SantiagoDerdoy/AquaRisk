# core/models.py

import numpy as np
from config import DECLINE_RATES

def trend_model(last_value, well, months):

    slope_monthly = DECLINE_RATES[well] / 12
    forecast = []

    level = last_value

    for _ in range(months):
        level = level + slope_monthly
        forecast.append(level)

    return np.array(forecast)

def hybrid_model(last_value, well, months,
                 rain_projection,
                 pump_projection,
                 et_projection,
                 rain_coeff,
                 pump_coeff,
                 et_coeff=0.002):

    slope_monthly = DECLINE_RATES[well] / 12
    forecast = []
    level = last_value

    for m in range(months):

        recharge = rain_coeff[well] * rain_projection[m]
        drawdown = pump_coeff[well] * (pump_projection[m] / 1e6)
        et_loss = et_coeff * et_projection[m]

        level = level + slope_monthly - recharge + drawdown + et_loss
        forecast.append(level)

    return np.array(forecast)

def smoothed_model(last_value, well, months):

    slope_monthly = (DECLINE_RATES[well] * 0.7) / 12

    forecast = []
    level = last_value

    for _ in range(months):
        level = level + slope_monthly
        forecast.append(level)

    return np.array(forecast)