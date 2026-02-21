# core/data_processing.py

import pandas as pd

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def validate_dataset(df):
    if df.isnull().sum().sum() > 0:
        df = df.interpolate()
    return df