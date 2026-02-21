# config.py

CLIENT_NAME = "Green Valley Agricultural Operations"
LOCATION = "Central Valley, California"

THRESHOLD_LEVEL = 15.0
FORECAST_HORIZON_MONTHS = 24
MONTE_CARLO_SIMULATIONS = 2000

DECLINE_RATES = {
    "GW-01": 0.9,
    "GW-02": 0.7,
    "GW-03": 1.3,
    "GW-04": 1.0
}

NOISE_SD = {
    "GW-01": 0.25,
    "GW-02": 0.3,
    "GW-03": 0.35,
    "GW-04": 0.4
}

# Hydrogeological coefficients

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