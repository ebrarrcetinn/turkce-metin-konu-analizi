import json

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.svm import LinearSVC

from src.evaluation.metrics import expected_calibration_error, multilabel_set_metrics
from src.models.artifact import ModelArtifactError, load_artifact, save_artifact
from src.models.calibration import TemperatureScaledClassifier, fit_temperature


def test_fit_temperature_recovers_true_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(4000, 5))
    true_t = 2.0
    y = np.array([rng.choice(5, p=p) for p in softmax(logits / true_t, axis=1)])
    assert fit_temperature(logits, y) == pytest.approx(true_t, rel=0.1)


class _VecLinear:
    """Metin listesini uzunluk tabanlı özelliklere çeviren küçük sahte tahminci."""

    def __init__(self):
        self.clf = LinearSVC(random_state=0)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    @staticmethod
    def _x(texts):
        return np.array([[len(t), t.count("a")] for t in texts], dtype=float)

    def fit(self, texts, y):
        self.clf.fit(self._x(texts), y)
        self.classes_ = self.clf.classes_
        return self

    def decision_function(self, texts):
        return self.clf.decision_function(self._x(texts))


def test_temperature_classifier_keeps_argmax_and_normalizes():
    texts = ["a" * i + "b" * (20 - i) for i in range(20)] * 5
    labels = ["az" if t.count("a") < 7 else "orta" if t.count("a") < 14 else "çok"
              for t in texts]
    model = TemperatureScaledClassifier(_VecLinear(), cv=3).fit(texts, labels)
    proba = model.predict_proba(texts)
    assert np.allclose(proba.sum(axis=1), 1.0)
    base_argmax = model.classes_[np.argmax(model.decision_function(texts), axis=1)]
    assert (model.predict(texts) == base_argmax).all()
    assert model.temperature_ > 0


def test_ece_bounds():
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert expected_calibration_error([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def test_multilabel_set_metrics():
    m = multilabel_set_metrics([{"a"}, {"b"}], [{"a"}, {"a", "b"}], ["a", "b"])
    assert m["subset_accuracy"] == 0.5
    assert m["recall_at_set"] == 1.0
    assert m["hamming_loss"] == 0.25


class _Toy:
    classes_ = np.array(["Diğer", "Fizik > Optik"])


def _meta():
    return {"model_name": "t", "model_version": "0", "training_timestamp": "x",
            "dataset": {"version": "x"}, "labels": {"joint": ["Diğer", "Fizik > Optik"]},
            "preprocessing": "x", "hyperparameters": {}, "thresholds": {}, "metrics": {}}


def test_artifact_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "m.joblib"
    meta = save_artifact(_Toy(), _meta(), path)
    model, loaded = load_artifact(path)
    assert loaded["artifact_sha256"] == meta["artifact_sha256"]
    assert list(model.classes_) == ["Diğer", "Fizik > Optik"]

    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ModelArtifactError, match="SHA-256"):
        load_artifact(path)


def test_artifact_missing_gives_instructions(tmp_path):
    with pytest.raises(ModelArtifactError, match="python train.py"):
        load_artifact(tmp_path / "yok.joblib")


def test_artifact_bad_metadata(tmp_path):
    path = tmp_path / "m.joblib"
    save_artifact(_Toy(), _meta(), path)
    path.with_suffix(".json").write_text("{bozuk", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="bozuk"):
        load_artifact(path)
    path.with_suffix(".json").write_text(json.dumps({"model_name": "t"}), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="eksik"):
        load_artifact(path)


def test_artifact_class_mismatch(tmp_path):
    path = tmp_path / "m.joblib"
    meta = _meta()
    meta["labels"]["joint"] = ["Başka"]
    save_artifact(_Toy(), meta, path)
    with pytest.raises(ModelArtifactError, match="sınıfları"):
        load_artifact(path)
