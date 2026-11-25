import pandas as pd

FEATURE_COLUMNS = [
    "hour", "minute", "day_of_week", "month", "day_of_year",
    "is_weekend", "flow_lag_1", "flow_lag_12", "flow_lag_288",
    "rolling_mean_12",
]


def prepare_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["flow"]).copy()

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Twelve records equal one hour; 288 records equal one day.
    df["flow_lag_1"] = df["flow"].shift(1)
    df["flow_lag_12"] = df["flow"].shift(12)
    df["flow_lag_288"] = df["flow"].shift(288)
    df["rolling_mean_12"] = df["flow"].shift(1).rolling(12).mean()

    return df.dropna(subset=FEATURE_COLUMNS + ["flow"]).copy()


def chronological_split(df, train_fraction=0.80):
    split_index = int(len(df) * train_fraction)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()