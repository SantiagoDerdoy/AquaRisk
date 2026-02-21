# core/ensemble.py

import numpy as np

def weighted_ensemble(hybrid, trend, smooth):

    w1 = 0.5
    w2 = 0.3
    w3 = 0.2

    ensemble_forecast = w1*hybrid + w2*trend + w3*smooth

    return ensemble_forecast