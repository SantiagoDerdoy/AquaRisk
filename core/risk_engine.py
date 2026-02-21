# core/risk_engine.py
import numpy as np

def calculate_time_to_threshold(simulations, threshold):

    crossing_times = []

    for sim in simulations:
        below = np.where(sim <= threshold)[0]

        if len(below) > 0:
            crossing_times.append(below[0])

    if len(crossing_times) == 0:
        return None, 0.0, 0.0

    crossing_times = np.array(crossing_times)

    mean_cross = np.mean(crossing_times)

    prob_12 = np.mean(crossing_times <= 12)
    prob_24 = np.mean(crossing_times <= 24)

    return mean_cross, prob_12, prob_24
    
def classify_risk(probability):
    if probability < 0.1:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    else:
        return "High Risk"