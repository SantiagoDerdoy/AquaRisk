# core/ensemble.py
# ─────────────────────────────────────────────
# AquaRisk — Ensemble Model Weighting
#
# The ensemble combines three models:
#   hybrid  (0) — physical-statistical, scenario-sensitive
#   trend   (1) — pure historical linear trend
#   smooth  (2) — dampened trend
#
# Scenario-aware weighting:
#   For stress scenarios (drought, expansion), the physical
#   hybrid model gets more weight because the scenario drivers
#   are the dominant force — pure trend under-captures them.
#   For baseline/sustainable, trend and smooth get more weight
#   since the historical pattern is the best predictor.
# ─────────────────────────────────────────────

import numpy as np


# Weights per scenario — must sum to 1.0
SCENARIO_WEIGHTS = {
    "drought":     (0.70, 0.20, 0.10),   # heavy hybrid — stress scenario dominates
    "expansion":   (0.70, 0.20, 0.10),   # heavy hybrid — pumping scenario dominates
    "baseline":    (0.45, 0.35, 0.20),   # balanced — historical trend matters most
    "sustainable": (0.50, 0.30, 0.20),   # moderate hybrid — conservation effect matters
}

DEFAULT_WEIGHTS = (0.50, 0.30, 0.20)


def weighted_ensemble(
    hybrid:   np.ndarray,
    trend:    np.ndarray,
    smooth:   np.ndarray,
    scenario: str = "baseline",
) -> np.ndarray:
    """
    Weighted ensemble of three forecast models.

    Parameters
    ----------
    hybrid   : scenario-sensitive hybrid physical model
    trend    : pure linear trend extrapolation
    smooth   : dampened trend (conservative)
    scenario : name of the active scenario (affects weights)

    Returns
    -------
    np.ndarray — weighted ensemble forecast
    """
    w1, w2, w3 = SCENARIO_WEIGHTS.get(scenario, DEFAULT_WEIGHTS)
    return w1 * hybrid + w2 * trend + w3 * smooth