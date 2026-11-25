"""Model evaluation helpers for the animal shelter model."""

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def evaluate(
    clf_model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, Any]:
    """Evaluate a fitted model on labelled data.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    features : pandas.DataFrame
        Feature data.
    target : pandas.Series
        True labels.

    Returns
    -------
    dict
        Accuracy, classification report and confusion matrix.

    """
    y_pred = clf_model.predict(features)
    labels = [str(label) for label in clf_model.classes_]
    raw_report = classification_report(
        target, y_pred, output_dict=True, zero_division=0
    )
    report = {
        key: (
            {name: float(value) for name, value in scores.items()}
            if isinstance(scores, dict)
            else float(scores)
        )
        for key, scores in raw_report.items()
    }
    matrix = confusion_matrix(target, y_pred, labels=clf_model.classes_)
    return {
        "accuracy": float(accuracy_score(target, y_pred)),
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
    }


def feature_importances(clf_model: Pipeline) -> dict[str, float]:
    """Get feature importances of the fitted model, sorted from high to low.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline with a ``transformer`` and ``model`` step.

    Returns
    -------
    dict[str, float]
        Mapping of transformed feature name to importance.

    """
    names = clf_model.named_steps["transformer"].get_feature_names_out()
    importances = clf_model.named_steps["model"].feature_importances_
    pairs = sorted(zip(names, importances, strict=True), key=lambda p: -p[1])
    return {str(name): float(value) for name, value in pairs}
