import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(**linear_regression_parameters):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(**linear_regression_parameters)),
    ])


def train_model(X_train, y_train, **linear_regression_parameters):
    model = build_model(**linear_regression_parameters)
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def get_coefficients(model, feature_names=None):
    estimator = model.named_steps["model"]
    coefficients = estimator.coef_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    return pd.DataFrame({
        "feature": list(feature_names),
        "coefficient": coefficients,
    }).sort_values("coefficient", key=abs, ascending=False)