# core/scenario_engine.py

import numpy as np


def generate_scenario(months, scenario_type):
    """
    Generates standardized climate & pumping scenarios.

    Units:
    - Rain: mm/month
    - Pump: million m3/month
    - ET: mm/month
    """

    if scenario_type == "baseline":
        rain = np.full(months, 50.0)
        pump = np.full(months, 5.0)
        et = np.full(months, 30.0)

    elif scenario_type == "drought":
        rain = np.full(months, 20.0)
        pump = np.full(months, 6.0)
        et = np.full(months, 40.0)

    elif scenario_type == "sustainable":
        rain = np.full(months, 60.0)
        pump = np.full(months, 4.0)
        et = np.full(months, 25.0)

    elif scenario_type == "expansion":
        rain = np.full(months, 45.0)
        pump = np.full(months, 8.0)
        et = np.full(months, 35.0)

    else:
        raise ValueError("Escenario no reconocido")

    return rain, pump, et