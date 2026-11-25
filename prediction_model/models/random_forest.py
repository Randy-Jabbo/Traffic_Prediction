from sklearn.ensemble import RandomForestRegressor


def build_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )