"""REST API for training the animal shelter model and making predictions."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from animal_shelter.model.predict import predict
from animal_shelter.model.train import metrics_path, train

app = FastAPI()

model = None
logger = logging.getLogger("animal_shelter")


class ConfusionMatrix(BaseModel):
    """Confusion matrix with row/column labels."""

    labels: list[str]
    matrix: list[list[int]]


class Metrics(BaseModel):
    """Evaluation metrics of a trained model."""

    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int
    classification_report: dict[str, dict[str, float] | float]
    confusion_matrix: ConfusionMatrix
    feature_importances: dict[str, float]
    trained_at: datetime


class TrainResponse(BaseModel):
    """Response of the train endpoint."""

    message: str
    metrics: Metrics


class PredictionItem(BaseModel):
    """Predicted outcome for a single animal."""

    id: str
    outcome_type: str


class PredictResponse(BaseModel):
    """Response of the predict endpoint."""

    n_predictions: int
    predictions: list[PredictionItem]
    class_distribution: dict[str, int]


class InputData(BaseModel):
    """Paths to the input data and the model file."""

    data_path: str
    model_path: str

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v):
        """Check that the data path points to a readable CSV file."""
        try:
            pd.read_csv(
                v, parse_dates=["DateTime"]
            )  # Check if the CSV file can be read
        except Exception as e:
            raise ValueError(f"Invalid data path: {v}") from e
        return v

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        """Check that the model path has a ``.pkl`` extension."""
        if not v.endswith(".pkl"):
            raise ValueError("Model path must end with .pkl")
        return v


@app.post("/train/", response_model=TrainResponse)
def train_model(data: InputData):
    """Train a model, save it and return its evaluation metrics."""
    # Load/process data
    model_file = Path(data.model_path)
    if model_file.exists():
        logger.info("Model already exists. Overwriting the existing model.")
        last_update = datetime.fromtimestamp(model_file.stat().st_mtime, tz=UTC)
    else:
        last_update = datetime.now(tz=UTC)

    _, metrics = train(data_path=data.data_path, model_path=data.model_path)
    # Pass data to functions from animal_shelter
    model = joblib.load(data.model_path)  # Load the trained model
    # Check that the model is saved correctly
    if model is None:
        logger.error("Model loading failed")
        raise ValueError("Model loading failed")
    elif last_update > datetime.fromtimestamp(model_file.stat().st_mtime, tz=UTC):
        logger.warning("Model file was not updated after training")
        raise ValueError("Model file was not updated after training")
    # Generate a response with useful information
    return TrainResponse(
        message="Model trained and loaded successfully",
        metrics=Metrics(**metrics),
    )


@app.get("/metrics/", response_model=Metrics)
def get_metrics(model_path: str):
    """Return the stored evaluation metrics of a trained model."""
    if not model_path.endswith(".pkl"):
        raise HTTPException(status_code=422, detail="Model path must end with .pkl")
    path = metrics_path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics found for this model")
    return Metrics(**json.loads(path.read_text()))


@app.post("/predict/", response_model=PredictResponse)
def predict_model(data: InputData):
    """Predict outcomes for the given data with a trained model."""
    # Load/process data
    model = joblib.load(data.model_path)
    if model is None:
        logger.error("Model loading failed")
        raise ValueError("Model loading failed")

    predictions = predict(data_path=data.data_path, model_path=data.model_path)
    # Generate a response with the predictions
    class_distribution = predictions["outcome_type"].value_counts()
    return PredictResponse(
        n_predictions=len(predictions),
        predictions=[
            PredictionItem(id=str(id_), outcome_type=str(outcome))
            for id_, outcome in zip(
                predictions["id"], predictions["outcome_type"], strict=True
            )
        ],
        class_distribution={str(k): int(v) for k, v in class_distribution.items()},
    )
