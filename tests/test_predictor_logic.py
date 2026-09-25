"""Model gerektirmeyen çıkarım mantığı testleri (hiyerarşi, güven, alt konu seçimi)."""

import json

import numpy as np
import pytest

from src.config import OTHER_LABEL, UNCERTAIN_LABEL
from src.models.predictor import (
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNCERTAIN,
    Prediction,
    aggregate,
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


def test_decide_per_class_thresholds():
    thresholds = {"Fizik": 0.5, "Bilim": 0.9}
    assert decide({"Fizik": 0.6, "Diğer": 0.4}, thresholds)[0] == STATUS_OK
    assert decide({"Bilim": 0.8, "Diğer": 0.2}, thresholds)[0] == STATUS_UNCERTAIN


def test_select_subtopics_threshold_and_limit():
    cond = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}
    assert [n for n, _ in select_subtopics(cond, 0.5)] == ["A"]
    assert [n for n, _ in select_subtopics(cond, 0.1)] == ["A", "B", "C"]
    assert [n for n, _ in select_subtopics(cond, 0.01, max_items=2)] == ["A", "B"]
    assert select_subtopics({}, 0.3) == []


def test_prediction_json_serialization():
    p = Prediction(text="x", status=STATUS_OK, general="Fizik", confidence=0.9,
                   general_scores={"Fizik": 0.9}, subtopics=[("Optik", 0.8)])
    payload = json.loads(json.dumps(p.to_dict(), ensure_ascii=False))
    assert payload["subtopics"] == [{"name": "Optik", "score": 0.8}]
    assert payload["general"] == "Fizik"
