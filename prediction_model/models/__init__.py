from .gradient_boosting import build_model as build_gradient_boosting
from .knn import build_model as build_knn
from .linear_regression import build_model as build_linear_regression
from .random_forest import build_model as build_random_forest

MODEL_BUILDERS = {
    "linear_regression": build_linear_regression,
    "gradient_boosting": build_gradient_boosting,
    "random_forest": build_random_forest,
    "knn": build_knn,
}


def build_model(model_name):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_BUILDERS[model_name]()


__all__ = ["MODEL_BUILDERS", "build_model"]