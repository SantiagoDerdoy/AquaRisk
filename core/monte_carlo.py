# core/monte_carlo.py
import numpy as np
from config import MONTE_CARLO_SIMULATIONS, NOISE_SD, THRESHOLD_LEVEL


def run_monte_carlo(forecast, well, threshold=None):
    """
    Monte Carlo simulation with temporal correlation (AR1 noise).

    Exceedance probability = probability that groundwater level falls
    BELOW the critical threshold at any point during the forecast horizon.
    (Lower level = higher risk for water supply.)

    Parameters
    ----------
    forecast  : np.ndarray — ensemble forecast array
    well      : str — well name (used to look up NOISE_SD)
    threshold : float, optional — critical level in meters.
                If None, uses THRESHOLD_LEVEL from config.
    """
    if threshold is None:
        threshold = THRESHOLD_LEVEL

    n_sim = MONTE_CARLO_SIMULATIONS
    phi   = 0.6  # AR1 temporal correlation

    noise_sd = NOISE_SD.get(well, 0.3)

    sims = []
    for _ in range(n_sim):
        noise      = np.zeros(len(forecast))
        base_noise = np.random.normal(0, noise_sd, len(forecast))
        for t in range(1, len(forecast)):
            noise[t] = phi * noise[t - 1] + base_noise[t]
        sims.append(forecast + noise)

    sims = np.array(sims)

    # Probability that level drops BELOW threshold at any point
    exceedances = np.sum(np.any(sims < threshold, axis=1))
    probability = float(exceedances / n_sim)

    p5  = np.percentile(sims,  5, axis=0)
    p95 = np.percentile(sims, 95, axis=0)

    return sims, probability, p5, p95