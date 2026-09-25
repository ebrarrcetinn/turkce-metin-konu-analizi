"""Değerlendirme metrikleri ve kalibrasyon ölçümleri."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true: Sequence, y_pred: Sequence,
                           labels: Sequence[str] | None = None) -> dict:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    kwargs = {"labels": labels, "zero_division": 0}
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kwargs)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kwargs)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kwargs)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", **kwargs)),
        "per_class": classification_report(y_true, y_pred, output_dict=True, **kwargs),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def summary(metrics: dict, digits: int = 4) -> dict:
    keys = ("n", "accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1")
    return {k: (round(metrics[k], digits) if isinstance(metrics[k], float) else metrics[k])
            for k in keys}


def _bins(confidences: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        yield lo, hi, (confidences > lo) & (confidences <= hi)


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """ECE: güven aralıklarında |doğruluk - ortalama güven| farkının ağırlıklı ortalaması."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    return float(sum(mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
                     for _, _, mask in _bins(conf, n_bins) if mask.any()))


def reliability_table(confidences, correct, n_bins: int = 10) -> list[dict]:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    return [{"bin": f"({lo:.1f},{hi:.1f}]", "n": int(mask.sum()),
             "mean_confidence": round(float(conf[mask].mean()), 3),
             "accuracy": round(float(corr[mask].mean()), 3)}
            for lo, hi, mask in _bins(conf, n_bins) if mask.any()]


def multilabel_set_metrics(true_sets: list[set[str]], pred_sets: list[set[str]],
                           labels: Sequence[str]) -> dict:
    """Küme tahminleri için micro/macro F1, Hamming loss, subset accuracy, P@1."""
    index = {label: i for i, label in enumerate(labels)}
    y_true = np.zeros((len(true_sets), len(labels)), dtype=int)
    y_pred = np.zeros_like(y_true)
    for row, (t, p) in enumerate(zip(true_sets, pred_sets, strict=True)):
        for label in t:
            y_true[row, index[label]] = 1
        for label in p:
            if label in index:
                y_pred[row, index[label]] = 1
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float((y_true != y_pred).mean()),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "recall_at_set": float(np.mean([bool(t & p) for t, p in
                                        zip(true_sets, pred_sets, strict=True)])),
        "mean_set_size": float(np.mean([len(p) for p in pred_sets])),
    }
