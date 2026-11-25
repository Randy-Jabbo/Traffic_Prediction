from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }
    return metrics, predictions