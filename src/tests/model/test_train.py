import json

from animal_shelter.model.train import metrics_path, train


def test_train_returns_and_saves_metrics(tmp_path):
    model_path = str(tmp_path / "model.pkl")

    _, metrics = train(data_path="data/train.csv", model_path=model_path)

    expected_keys = {
        "train_accuracy",
        "test_accuracy",
        "n_train",
        "n_test",
        "classification_report",
        "confusion_matrix",
        "feature_importances",
        "trained_at",
    }
    assert expected_keys <= metrics.keys()
    assert 0 <= metrics["train_accuracy"] <= 1
    assert 0 <= metrics["test_accuracy"] <= 1
    assert metrics["n_test"] > 0

    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    assert len(matrix) == len(labels)
    assert all(len(row) == len(labels) for row in matrix)
    assert sum(map(sum, matrix)) == metrics["n_test"]

    saved = json.loads(metrics_path(model_path).read_text())
    assert saved == metrics
