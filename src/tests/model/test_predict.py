import pandas as pd

from animal_shelter.model.predict import predict
from animal_shelter.model.train import train


def test_predict_returns_id_and_outcome(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    train(data_path="data/train.csv", model_path=model_path)

    predictions = predict(data_path="data/test.csv", model_path=model_path)

    assert list(predictions.columns) == ["id", "outcome_type"]
    assert len(predictions) == len(pd.read_csv("data/test.csv"))
