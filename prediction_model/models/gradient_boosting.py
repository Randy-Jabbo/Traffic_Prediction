from sklearn.ensemble import GradientBoostingRegressor


def build_model():
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )