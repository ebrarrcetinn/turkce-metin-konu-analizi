"""
Dosya   : tests/test_topic_model.py
Konu    : Konu Modeli Testleri
Açıklama: Olasılık toplama, karar verme, alt konu seçimi ve model dosyasının kaydedilip
          yüklenmesini test eder (eğitilmiş model gerektirmez).
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import json

import numpy as np
import pytest

from src.config import OTHER_LABEL, UNCERTAIN_LABEL
from src.models.topic_model import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNCERTAIN,
    ModelFileError,
    Prediction,
    TopicModel,
    aggregate,
    build_vectorizer,
    decide,
    joint_label,
    select_subtopics,
    split_joint,
)

CLASSES = ["Diğer", "Fizik > Kuantum Mekaniği", "Fizik > Optik",
           "Teknoloji > Kuantum Bilgisayarlar"]


def test_joint_label_roundtrip():
    assert split_joint(joint_label("Fizik", "Optik")) == ("Fizik", "Optik")
    assert split_joint(joint_label(OTHER_LABEL, None)) == (OTHER_LABEL, None)


def test_aggregate_marginals_and_conditionals():
    general, cond = aggregate(CLASSES, np.array([0.1, 0.3, 0.2, 0.4]))
    assert general == pytest.approx({"Diğer": 0.1, "Fizik": 0.5, "Teknoloji": 0.4})
    assert sum(general.values()) == pytest.approx(1.0)
    assert cond["Fizik"] == pytest.approx({"Kuantum Mekaniği": 0.6, "Optik": 0.4})
    assert OTHER_LABEL not in cond


def test_decide_statuses():
    assert decide({"Fizik": 0.9, "Diğer": 0.1}, 0.75) == (STATUS_OK, "Fizik", 0.9)
    assert decide({"Fizik": 0.6, "Diğer": 0.4}, 0.75) == (STATUS_UNCERTAIN, UNCERTAIN_LABEL, 0.6)
    assert decide({"Fizik": 0.3, "Diğer": 0.7}, 0.75)[0] == STATUS_OUT_OF_SCOPE


def test_select_subtopics_min_score_and_limit():
    cond = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}
    assert [n for n, _ in select_subtopics(cond, 0.5)] == ["A"]
    assert [n for n, _ in select_subtopics(cond, 0.1)] == ["A", "B", "C"]
    assert [n for n, _ in select_subtopics(cond, 0.01, limit=2)] == ["A", "B"]
    assert select_subtopics({}, 0.3) == []


def test_prediction_json_serialization():
    p = Prediction(text="x", status=STATUS_OK, general="Fizik", confidence=0.9,
                   general_scores={"Fizik": 0.9}, subtopics=[("Optik", 0.8)],
                   related_topics=[("Teknoloji", 0.3)])
    payload = json.loads(json.dumps(p.to_dict(), ensure_ascii=False))
    assert payload["subtopics"] == [{"name": "Optik", "score": 0.8}]
    assert payload["related_topics"] == [{"name": "Teknoloji", "score": 0.3}]


@pytest.fixture
def tiny_model():
    texts = ["kuantum dolanıklık dalga parçacık", "kuantum kübit işlemci algoritma",
             "ışık mercek kırılma optik", "futbol maç gol hakem",
             "kuantum dalga fonksiyonu parçacık", "kübit kuantum bilgisayar devre"]
    labels = ["Fizik > Kuantum Mekaniği", "Teknoloji > Kuantum Bilgisayarlar", "Fizik > Optik",
              "Diğer", "Fizik > Kuantum Mekaniği", "Teknoloji > Kuantum Bilgisayarlar"]
    vectorizer = build_vectorizer()
    features = vectorizer.fit_transform(texts)
    classes = sorted(set(labels))
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(features.shape[1], len(classes))).astype(np.float32)
    bias = np.zeros(len(classes), dtype=np.float32)
    return TopicModel(vectorizer, classes, weights, bias, 1.3, 0.4,
                      {"model_version": "test"})


def test_save_and_load_roundtrip(tiny_model, tmp_path):
    path = tmp_path / "model"
    tiny_model.save(path)
    loaded = TopicModel.load(path)
    text = "kuantum kübit"
    assert loaded.classes == tiny_model.classes and loaded.version == "test"
    assert loaded.temperature == pytest.approx(1.3)
    np.testing.assert_allclose(loaded.predict_proba([text]), tiny_model.predict_proba([text]),
                               rtol=1e-5)


def test_load_rejects_missing_and_incompatible_files(tiny_model, tmp_path):
    with pytest.raises(ModelFileError):
        TopicModel.load(tmp_path / "yok")
    path = tmp_path / "model"
    tiny_model.save(path)
    config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    config["format_version"] = 999
    path.with_suffix(".json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelFileError):
        TopicModel.load(path)


def test_predict_empty_text(tiny_model):
    assert tiny_model.predict("!!! ...").status == STATUS_EMPTY


def test_prediction_probabilities_sum_to_one(tiny_model):
    p = tiny_model.predict("kuantum parçacık")
    assert sum(p.general_scores.values()) == pytest.approx(1.0, abs=1e-3)
