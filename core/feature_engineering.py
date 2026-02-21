# core/feature_engineering.py

import numpy as np

def calculate_rate_of_decline(series):
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    return slope


def seasonal_amplitude(series):
    return series.max() - series.min()