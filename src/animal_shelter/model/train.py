"""Training of the animal shelter outcome model."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model.evaluate import evaluate, feature_importances


def train(
    data_path: str,
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a model on the animal shelter data and evaluate it on a hold-out set.

    Parameters
    ----------
    data_path : str
        Path to the training data
    model_path : str
        Path to save the model
    test_size : float
        Fraction of the data used as stratified hold-out test set
    random_state : int
        Random seed for the split and the model

    Returns
    -------
    tuple[sklearn.pipeline.Pipeline, dict]
        A trained model pipeline and its evaluation metrics

    """
    raw_data = load_data(data_path)
    with_features = add_features(raw_data)
    cat_features = [
        "animal_type",
        "is_dog",
        "has_name",
        "sex",
        "hair_type",
    ]
    num_features = ["days_upon_outcome"]
    train_data, test_data = train_test_split(
        with_features,
        test_size=test_size,
        stratify=with_features["outcome_type"],
        random_state=random_state,
    )
    clf_model = _build_pipeline(cat_features, num_features, random_state)
    clf_model = _fit_model(clf_model, train_data, cat_features, num_features)

    feature_columns = [*cat_features, *num_features]
    train_metrics = evaluate(
        clf_model, train_data[feature_columns], train_data["outcome_type"]
    )
    test_metrics = evaluate(
        clf_model, test_data[feature_columns], test_data["outcome_type"]
    )
    metrics = {
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "n_train": len(train_data),
        "n_test": len(test_data),
        "classification_report": test_metrics["classification_report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "feature_importances": feature_importances(clf_model),
        "trained_at": datetime.now(tz=UTC).isoformat(),
    }

    _save_model(clf_model, model_path)
    _save_metrics(metrics, model_path)
    return clf_model, metrics


def _build_pipeline(
    cat_features: Sequence[str],
    num_features: Sequence[str],
    random_state: int | None = None,
) -> Pipeline:
    """Build the preprocessing and classification pipeline.

    Parameters
    ----------
    cat_features : Sequence[str]
        Categorical feature names.
    num_features : Sequence[str]
        Numeric feature names.
    random_state : int or None
        Random seed for the classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted model pipeline.

    """
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])
    transformer = ColumnTransformer(
        (
            ("numeric", num_transformer, num_features),
            ("categorical", cat_transformer, cat_features),
        )
    )
    clf_model = Pipeline(
        [
            ("transformer", transformer),
            ("model", RandomForestClassifier(random_state=random_state)),
        ]
    )
    return clf_model


def _fit_model(
    clf_model: Pipeline,
    with_features: pd.DataFrame,
    cat_features: Sequence[str],
    num_features: Sequence[str],
) -> Pipeline:
    """Fit the model pipeline using selected features.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        Model pipeline to fit.
    with_features : pandas.DataFrame
        Training data with engineered features and target.
    cat_features : Sequence[str]
        Categorical feature names.
    num_features : Sequence[str]
        Numeric feature names.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted model pipeline.

    """
    feature_columns = [*cat_features, *num_features]
    X = with_features[feature_columns]
    y = with_features["outcome_type"]
    clf_model.fit(X, y)
    return clf_model


def _save_model(clf_model: Pipeline, model_path: str) -> None:
    """Save the model to a file.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        A trained model pipeline
    model_path : str
        Path to save the model

    """
    joblib.dump(clf_model, model_path)


def metrics_path(model_path: str) -> Path:
    """Return the path of the metrics file belonging to a model.

    Parameters
    ----------
    model_path : str
        Path of the model file

    Returns
    -------
    pathlib.Path
        Path of the metrics JSON file next to the model

    """
    return Path(model_path).with_suffix(".metrics.json")


def _save_metrics(metrics: dict[str, Any], model_path: str) -> None:
    """Save the evaluation metrics as JSON next to the model.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics
    model_path : str
        Path of the model file

    """
    metrics_path(model_path).write_text(json.dumps(metrics, indent=2))
