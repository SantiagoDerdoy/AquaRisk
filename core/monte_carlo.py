# core/monte_carlo.py

import numpy as np
from config import MONTE_CARLO_SIMULATIONS, NOISE_SD, THRESHOLD_LEVEL

def run_monte_carlo(forecast, well):

    n_sim = MONTE_CARLO_SIMULATIONS

    sims = []

    for _ in range(n_sim):
        noise = np.random.normal(0, NOISE_SD[well], len(forecast))
        simulated = forecast + noise
        sims.append(simulated)

    sims = np.array(sims)

    exceedances = np.sum(np.any(sims > THRESHOLD_LEVEL, axis=1))
    probability = exceedances / n_sim

    p5 = np.percentile(sims, 5, axis=0)
    p95 = np.percentile(sims, 95, axis=0)

    return sims, probability, p5, p95