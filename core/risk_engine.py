# core/risk_engine.py
import numpy as np
from config import THRESHOLD_LEVEL


def calculate_time_to_threshold(simulations, threshold=None):
    """
    Calculate mean crossing time and breach probabilities.

    Parameters
    ----------
    simulations : np.ndarray shape (n_sims, n_months)
    threshold   : float, optional — uses config default if None

    Returns
    -------
    mean_cross : float | None — average month of first breach
    prob_12    : float — probability of breach within 12 months
    prob_24    : float — probability of breach within 24 months
    """
    if threshold is None:
        threshold = THRESHOLD_LEVEL

    crossing_times = []
    for sim in simulations:
        below = np.where(sim <= threshold)[0]
        if len(below) > 0:
            crossing_times.append(below[0])

    if len(crossing_times) == 0:
        return None, 0.0, 0.0

    crossing_times = np.array(crossing_times)
    mean_cross = float(np.mean(crossing_times))
    prob_12    = float(np.mean(crossing_times <= 12))
    prob_24    = float(np.mean(crossing_times <= 24))

    return mean_cross, prob_12, prob_24


def classify_risk(probability):
    """
    Classify risk level based on exceedance probability.

    Thresholds:
        < 15%  → Low Risk
        < 40%  → Moderate Risk
        < 70%  → High Risk
        >= 70% → Critical Risk
    """
    if probability < 0.15:
        return "Low Risk"
    elif probability < 0.40:
        return "Moderate Risk"
    elif probability < 0.70:
        return "High Risk"
    else:
        return "Critical Risk"