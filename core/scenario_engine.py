# core/scenario_engine.py

import numpy as np

def generate_scenario(months, scenario_type):

    if scenario_type == "baseline":
        rain = np.full(months, 50)
        pump = np.full(months, 5e6)
        et = np.full(months, 30)

    elif scenario_type == "drought":
        rain = np.full(months, 20)
        pump = np.full(months, 6e6)
        et = np.full(months, 40)

    elif scenario_type == "sustainable":
        rain = np.full(months, 60)
        pump = np.full(months, 4e6)
        et = np.full(months, 25)

    elif scenario_type == "expansion":
        rain = np.full(months, 45)
        pump = np.full(months, 8e6)
        et = np.full(months, 35)

    else:
        raise ValueError("Escenario no reconocido")

    return rain, pump, et