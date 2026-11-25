"""Prediction with a trained animal shelter model."""

import joblib
import pandas as pd

from animal_shelter.data import load_data
from animal_shelter.features import add_features


def predict(data_path: str, model_path: str) -> pd.DataFrame:
    """Predict the outcome type for the animal shelter data.

    Parameters
    ----------
    data_path : str
        Path to the test data
    model_path : str
        Path to the trained model

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` and ``outcome_type`` with one row per animal

    """
    cat_features = [
        "animal_type",
        "is_dog",
        "has_name",
        "sex",
        "hair_type",
    ]
    num_features = ["days_upon_outcome"]
    test_data = load_data(data_path)
    with_features = add_features(test_data)
    X_test = with_features[cat_features + num_features]
    clf_model = joblib.load(model_path)

    if "id" in with_features.columns:
        ids = with_features["id"]
    elif "animal_id" in with_features.columns:
        ids = with_features["animal_id"]
    else:
        ids = pd.Series(with_features.index)
    return pd.DataFrame(
        {"id": ids.astype(str).to_numpy(), "outcome_type": clf_model.predict(X_test)}
    )
