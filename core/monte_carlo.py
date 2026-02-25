# core/monte_carlo.py
import numpy as np
from config import MONTE_CARLO_SIMULATIONS, NOISE_SD, THRESHOLD_LEVEL


def run_monte_carlo(forecast, well):
    """
    Monte Carlo simulation with temporal correlation (AR1 noise).

    Exceedance probability = probability that groundwater level falls
    BELOW the critical threshold at any point during the forecast horizon.
    (Lower level = higher risk for water supply.)
    """
    n_sim = MONTE_CARLO_SIMULATIONS
    phi = 0.6  # AR1 temporal correlation

    sims = []

    for _ in range(n_sim):
        noise = np.zeros(len(forecast))
        base_noise = np.random.normal(0, NOISE_SD[well], len(forecast))
        for t in range(1, len(forecast)):
            noise[t] = phi * noise[t - 1] + base_noise[t]
        simulated = forecast + noise
        sims.append(simulated)

    sims = np.array(sims)

    # Risk = any simulation touches or crosses BELOW threshold
    exceedances = np.sum(np.any(sims < THRESHOLD_LEVEL, axis=1))
    probability = exceedances / n_sim

    p5 = np.percentile(sims, 5, axis=0)
    p95 = np.percentile(sims, 95, axis=0)

    return sims, probability, p5, p95