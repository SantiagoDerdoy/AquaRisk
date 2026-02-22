# core/acei_engine.py

import numpy as np


def _normalize(value, min_val, max_val):
    """
    Normaliza un valor a escala 0-100.
    """
    if max_val - min_val == 0:
        return 0
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized * 100, 0, 100)


def calculate_acei(
    probability,
    decline_rate,
    distance_to_threshold,
    volatility,
    prob_range=(0, 1),
    decline_range=(0, 5),
    distance_range=(0, 20),
    volatility_range=(0, 10)
):

    # 🔒 Forzar escalares reales
    probability = float(np.asarray(probability).mean())
    decline_rate = float(np.asarray(decline_rate).mean())
    distance_to_threshold = float(np.asarray(distance_to_threshold).mean())
    volatility = float(np.asarray(volatility).mean())

    # Normalización
    prob_score = _normalize(probability, *prob_range)
    decline_score = _normalize(abs(decline_rate), *decline_range)

    proximity_score = 100 - _normalize(distance_to_threshold, *distance_range)
    volatility_score = _normalize(volatility, *volatility_range)

    acei_score = (
        0.40 * prob_score +
        0.25 * decline_score +
        0.20 * proximity_score +
        0.15 * volatility_score
    )

    acei_score = round(float(np.asarray(acei_score)), 2)

    category, recommendation = _classify_acei(acei_score)

    return acei_score, category, recommendation


def _classify_acei(score):

    if score <= 20:
        return (
            "Secure",
            "Operational posture can be maintained. Continue standard monitoring."
        )

    elif score <= 40:
        return (
            "Stable",
            "No immediate intervention required. Maintain pumping strategy with quarterly review."
        )

    elif score <= 60:
        return (
            "Moderate Exposure",
            "Strategic adjustment recommended. Evaluate controlled pumping reduction and contingency planning."
        )

    elif score <= 80:
        return (
            "High Exposure",
            "Structured intervention required. Implement staged pumping optimization within 3–6 months."
        )

    else:
        return (
            "Critical Structural Risk",
            "Immediate action required. Capital allocation review and emergency water strategy advised."
        )


def calculate_portfolio_acei(well_scores, weights=None):
    """
    Calcula ACEI a nivel portfolio.
    well_scores: lista de scores individuales
    weights: ponderación opcional por pozo
    """

    if not well_scores:
        return 0, "Secure", "No data available."

    if weights and len(weights) == len(well_scores):
        portfolio_score = np.average(well_scores, weights=weights)
    else:
        portfolio_score = np.mean(well_scores)

    portfolio_score = round(float(portfolio_score), 2)

    category, recommendation = _classify_acei(portfolio_score)

    return portfolio_score, category, recommendation