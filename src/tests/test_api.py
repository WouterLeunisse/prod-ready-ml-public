from api.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_train_invalid_data_path():
    response = client.post(
        "/train/",
        json={"data_path": "invalid/path.csv", "model_path": "model.pkl"},
    )
    assert response.status_code == 422


def test_train_invalid_model_path():
    response = client.post(
        "/train/",
        json={"data_path": "data/train.csv", "model_path": "invalid_model.txt"},
    )
    assert response.status_code == 422


def test_train_valid_paths(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    response = client.post(
        "/train/",
        json={"data_path": "data/train.csv", "model_path": model_path},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Model trained and loaded successfully"
    assert 0 <= body["metrics"]["test_accuracy"] <= 1


def test_metrics_after_train(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    client.post(
        "/train/",
        json={"data_path": "data/train.csv", "model_path": model_path},
    )
    response = client.get("/metrics/", params={"model_path": model_path})
    assert response.status_code == 200
    body = response.json()
    assert {"train_accuracy", "test_accuracy", "confusion_matrix"} <= body.keys()


def test_metrics_missing(tmp_path):
    response = client.get(
        "/metrics/", params={"model_path": str(tmp_path / "missing.pkl")}
    )
    assert response.status_code == 404


def test_predict_returns_predictions(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    client.post(
        "/train/",
        json={"data_path": "data/train.csv", "model_path": model_path},
    )
    response = client.post(
        "/predict/",
        json={"data_path": "data/test.csv", "model_path": model_path},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_predictions"] == len(body["predictions"])
    assert sum(body["class_distribution"].values()) == body["n_predictions"]
