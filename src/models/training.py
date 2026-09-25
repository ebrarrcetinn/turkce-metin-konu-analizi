"""Final modelin eğitim hattı: kalibrasyon seçimi, eşik seçimi (validation), yeniden eğitim.

Test seti bu modülde hiç okunmaz.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn

from src.config import (
    CONFIDENCE_GRID,
    FINAL_C,
    FINAL_EXPERIMENT_ID,
    MODEL_PATH,
    MODEL_VERSION,
    OTHER_LABEL,
    RANDOM_SEED,
    RAW_DIR,
    REPORTS_DIR,
    THRESHOLD_GRID,
)
from src.data.build import load_split
from src.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    multilabel_set_metrics,
    reliability_table,
    summary,
)
from src.models.artifact import save_artifact
from src.models.calibration import fit_temperature
from src.models.pipeline import CALIBRATION_METHODS, build_calibrated, get_candidate
from src.models.predictor import aggregate, decide, joint_label, select_subtopics
from src.models.taxonomy import ALL_GENERAL_LABELS, subtopics_of

logger = logging.getLogger(__name__)

MAX_F1_LOSS_FOR_CALIBRATION = 0.005


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def joint_labels(frame: pd.DataFrame) -> list[str]:
    return [joint_label(g, s) for g, s in zip(frame["general"], frame["subtopic"], strict=True)]


def general_outputs(model, texts: list[str]) -> tuple[list[dict], list[dict]]:
    classes = [str(c) for c in model.classes_]
    out_general, out_cond = [], []
    for row in model.predict_proba(texts):
        general, conditional = aggregate(classes, row)
        out_general.append(general)
        out_cond.append(conditional)
    return out_general, out_cond


def argmax_general(scores: list[dict]) -> tuple[list[str], np.ndarray]:
    labels = [max(s, key=lambda k: s[k]) for s in scores]
    conf = np.array([s[label] for s, label in zip(scores, labels, strict=True)])
    return labels, conf


def predictions_with_threshold(scores: list[dict],
                               min_confidence: float | dict[str, float]) -> list[str]:
    """Belirsiz ve taksonomi dışı kararlar değerlendirmede 'Diğer' sayılır."""
    return [label if status == "ok" else OTHER_LABEL
            for status, label, _ in (decide(s, min_confidence) for s in scores)]


def choose_min_confidence(scores: list[dict], y_true: list[str]) -> tuple[float, list[dict]]:
    rows = []
    for t in CONFIDENCE_GRID:
        m = classification_metrics(y_true, predictions_with_threshold(scores, t),
                                   list(ALL_GENERAL_LABELS))
        rows.append({"threshold": t, "macro_f1": round(m["macro_f1"], 4),
                     "accuracy": round(m["accuracy"], 4),
                     "precision_macro": round(m["precision_macro"], 4),
                     "recall_macro": round(m["recall_macro"], 4)})
    best = max(rows, key=lambda r: (r["macro_f1"], -r["threshold"]))
    return best["threshold"], rows


def _macro_f1(y_true: list[str], scores: list[dict], thresholds) -> float:
    return classification_metrics(y_true, predictions_with_threshold(scores, thresholds),
                                  list(ALL_GENERAL_LABELS))["macro_f1"]


def choose_per_class_thresholds(scores: list[dict], y_true: list[str], start: float,
                                passes: int = 2) -> dict[str, float]:
    """Genel konu başına eşik; koordinat artışıyla macro F1'i maksimize eder."""
    thresholds = {label: start for label in ALL_GENERAL_LABELS if label != OTHER_LABEL}
    best = _macro_f1(y_true, scores, thresholds)
    for _ in range(passes):
        for label in thresholds:
            for t in CONFIDENCE_GRID:
                trial = {**thresholds, label: t}
                score = _macro_f1(y_true, scores, trial)
                if score > best + 1e-9:
                    best, thresholds = score, trial
    return thresholds


def choose_subtopic_threshold(conditionals: list[dict], frame: pd.DataFrame,
                              predicted_general: list[str]) -> tuple[float, list[dict]]:
    """Genel konu doğru tahmin edilen, taksonomi içi validation örneklerinde seçilir."""
    mask = [(g == p and s is not None) for g, p, s in
            zip(frame["general"], predicted_general, frame["subtopic"], strict=True)]
    all_subs = sorted({s for g in ALL_GENERAL_LABELS for s in subtopics_of(g)})
    rows = []
    for t in THRESHOLD_GRID:
        true_sets, pred_sets = [], []
        for keep, cond, g, s in zip(mask, conditionals, frame["general"], frame["subtopic"],
                                    strict=True):
            if keep:
                true_sets.append({s})
                pred_sets.append({name for name, _ in select_subtopics(cond.get(g, {}), t)})
        m = multilabel_set_metrics(true_sets, pred_sets, all_subs)
        rows.append({"threshold": t, **{k: round(v, 4) for k, v in m.items()}})
    best = max(rows, key=lambda r: (r["macro_f1"], -r["threshold"]))
    return best["threshold"], rows


def _provenance() -> dict:
    path = RAW_DIR / "provenance.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: {"commit": v["commit"], "license": v["license"], "homepage": v["homepage"],
                "downloaded_at_utc": v["downloaded_at_utc"]} for k, v in data.items()}


def _fit(method: str, frame: pd.DataFrame, store_oof: bool = False):
    model = build_calibrated(FINAL_EXPERIMENT_ID, method, C=FINAL_C)
    if method == "temperature":
        model.set_params(store_oof=store_oof)
        # Aynı terimin varyasyonları farklı OOF katlarına düşmesin.
        return model.fit(frame["text"].tolist(), joint_labels(frame), groups=frame["group"])
    return model.fit(frame["text"].tolist(), joint_labels(frame))


def train_final(model_path=MODEL_PATH) -> dict:
    seed_everything()
    train, val = load_split("train"), load_split("val")
    y_val = val["general"].tolist()
    labels = list(ALL_GENERAL_LABELS)

    # Kalibrasyon yöntemi: train ile eğitilip validation'da karşılaştırılır.
    calibration_report: dict = {}
    fitted: dict = {}
    for method in CALIBRATION_METHODS:
        t0 = time.perf_counter()
        model = _fit(method, train, store_oof=True)
        fit_s = time.perf_counter() - t0
        scores, conds = general_outputs(model, val["text"].tolist())
        pred, conf = argmax_general(scores)
        correct = np.array(pred) == np.array(y_val)
        m = classification_metrics(y_val, pred, labels)
        calibration_report[method] = {
            "val": summary(m),
            "ece_15bin": round(expected_calibration_error(conf, correct), 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "reliability": reliability_table(conf, correct),
            "fit_seconds": round(fit_s, 1),
        }
        fitted[method] = (model, scores, conds, pred, conf, correct)
        logger.info("Kalibrasyon %s: macroF1=%.4f ECE=%.4f", method, m["macro_f1"],
                    calibration_report[method]["ece_15bin"])

    best_f1 = max(r["val"]["macro_f1"] for r in calibration_report.values())
    eligible = [k for k, r in calibration_report.items()
                if r["val"]["macro_f1"] >= best_f1 - MAX_F1_LOSS_FOR_CALIBRATION]
    method = min(eligible, key=lambda k: calibration_report[k]["ece_15bin"])
    model, scores, conds, pred, conf, correct = fitted[method]

    # Güven eşiği
    min_conf, min_conf_rows = choose_min_confidence(scores, y_val)
    threshold_report: dict = {"global_val_tuned": {
        "threshold": min_conf, "val_macro_f1": round(_macro_f1(y_val, scores, min_conf), 4)}}
    per_class = None
    if hasattr(model, "oof_proba_"):
        # Sınıf bazlı eşik alternatifi: train OOF üzerinde seçilir, validation'da karşılaştırılır.
        classes = [str(c) for c in model.classes_]
        oof_scores = [aggregate(classes, row)[0] for row in model.oof_proba_]
        y_train = train["general"].tolist()
        global_oof, _ = choose_min_confidence(oof_scores, y_train)
        per_class_oof = choose_per_class_thresholds(oof_scores, y_train, global_oof)
        val_global = _macro_f1(y_val, scores, global_oof)
        val_per_class = _macro_f1(y_val, scores, per_class_oof)
        threshold_report["global_oof_tuned"] = {"threshold": global_oof,
                                                "val_macro_f1": round(val_global, 4)}
        threshold_report["per_class_oof_tuned"] = {"thresholds": per_class_oof,
                                                   "val_macro_f1": round(val_per_class, 4)}
        if val_per_class > max(val_global, threshold_report["global_val_tuned"]["val_macro_f1"]):
            per_class = per_class_oof
        for fitted_model, *_ in fitted.values():
            if hasattr(fitted_model, "oof_proba_"):
                del fitted_model.oof_proba_
    threshold_report["selected"] = "per_class_oof_tuned" if per_class else "global_val_tuned"
    active_thresholds = per_class or min_conf
    sub_thr, sub_rows = choose_subtopic_threshold(conds, val, pred)

    # Kısa girdiler için ayrı sıcaklık (validation terimleri, 1-3 sözcük)
    short_report = None
    short_t = None
    if method == "temperature":
        terms = val["term"].tolist()
        y_terms = np.searchsorted(model.classes_, joint_labels(val))
        short_t = fit_temperature(model.decision_function(terms), y_terms)
        short_report = {}
        for name, t in (("global_T", model.temperature_), ("short_T", short_t)):
            sc = [aggregate([str(c) for c in model.classes_], row)[0]
                  for row in model.predict_proba(terms, temperature=t)]
            p_, c_ = argmax_general(sc)
            corr_ = np.array(p_) == np.array(y_val)
            short_report[name] = {"temperature": round(float(t), 4),
                                  "accuracy": round(float(corr_.mean()), 4),
                                  "mean_confidence": round(float(c_.mean()), 4),
                                  "ece_15bin": round(expected_calibration_error(c_, corr_), 4)}
    val_thresholded = classification_metrics(
        y_val, predictions_with_threshold(scores, active_thresholds), labels)

    # Final model train + validation ile yeniden eğitilir.
    full = pd.concat([train, val], ignore_index=True)
    t0 = time.perf_counter()
    final = _fit(method, full)
    final_fit_s = time.perf_counter() - t0

    candidate = get_candidate(FINAL_EXPERIMENT_ID)
    metadata = {
        "model_name": f"turkish-topic-{FINAL_EXPERIMENT_ID.lower()}-flat-calibrated",
        "model_version": MODEL_VERSION,
        "training_timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "version": "taboo@f621b46+split-v1",
            "sources": _provenance(),
            "train_rows_final": int(len(full)),
            "train_rows_selection": int(len(train)),
            "val_rows": int(len(val)),
        },
        "labels": {
            "general": labels,
            "joint": [str(c) for c in final.classes_],
            "subtopics": {g: subtopics_of(g) for g in labels if subtopics_of(g)},
        },
        "architecture": {
            "experiment_id": FINAL_EXPERIMENT_ID,
            "description": candidate.name + " — düz 'Genel > Alt' sınıflandırma, genel konu "
                           "olasılığı marjinal toplam",
            "features": candidate.features,
        },
        "preprocessing": ("NFC + mojibake onarımı + HTML/URL/mention temizliği + Türkçe küçük "
                          "harf (I→ı, İ→i) + rakam/noktalama/emoji temizliği; word yolu F5 kök "
                          "+ stopword, char yolu char_wb 2-5gram"),
        "hyperparameters": {"C": FINAL_C, "class_weight": "balanced", "calibration": method,
                            "calibration_cv": 3, "random_seed": RANDOM_SEED,
                            "temperature": round(float(getattr(final, "temperature_", 0.0)), 4)
                            or None},
        "thresholds": {"min_confidence": min_conf, "subtopic_threshold": sub_thr,
                       "per_class_min_confidence": per_class,
                       "short_input_temperature": round(float(short_t), 4) if short_t else None},
        "metrics": {
            "validation_argmax": calibration_report[method]["val"],
            "validation_with_threshold": summary(val_thresholded),
            "validation_ece": calibration_report[method]["ece_15bin"],
            "note": "Validation metrikleri yalnızca train ile eğitilmiş modele aittir; "
                    "kaydedilen model train+val ile yeniden eğitilmiştir. Test metrikleri "
                    "evaluate.py tarafından eklenir.",
        },
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "final_fit_seconds": round(final_fit_s, 1),
    }
    metadata = save_artifact(final, metadata, model_path)

    report = {"selected_calibration": method, "calibration": calibration_report,
              "min_confidence_sweep": min_conf_rows, "subtopic_threshold_sweep": sub_rows,
              "threshold_selection": threshold_report,
              "short_input_calibration_val_terms": short_report,
              "thresholds": metadata["thresholds"],
              "val_per_class_f1_with_threshold": {
                  c: round(val_thresholded["per_class"][c]["f1-score"], 4) for c in labels}}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metadata": metadata, "report": report,
            "val_confidences": {"conf": conf.tolist(), "correct": correct.tolist()}}
