"""Kaydedilmiş modeli test, OOD ve harici setlerde değerlendirir ve raporları yazar."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, MODEL_PATH, OTHER_LABEL, REPORTS_DIR  # noqa: E402
from src.data.build import DatasetError, load_split  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    reliability_table,
    summary,
)
from src.logging_setup import setup_logging  # noqa: E402
from src.models.artifact import ModelArtifactError, write_metadata  # noqa: E402
from src.models.predictor import (  # noqa: E402
    STATUS_OK,
    TopicClassifier,
    aggregate,
    decide,
    select_subtopics,
)
from src.models.taxonomy import ALL_GENERAL_LABELS  # noqa: E402
from src.preprocessing.text import normalize  # noqa: E402

LABELS = list(ALL_GENERAL_LABELS)
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6e7781", "#2f6db3", "#d9822b"

# Bu cümleler eğitim verisinde bulunmaz.
QUANTUM_TESTS = [
    ("Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir.",
     "Fizik", "Kuantum Mekaniği"),
    ("Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir.",
     "Teknoloji", "Kuantum Bilgisayarlar"),
]
ACCEPTANCE_TESTS = [
    ("Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı.", "Kitaplar", None),
    ("Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder.", "Bilim", None),
    ("Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir.", "Biyoloji", None),
]


def batch_outputs(clf: TopicClassifier, texts: list[str]):
    proba = clf.predict_proba_auto(texts)
    rows = []
    for text, p in zip(texts, proba, strict=True):
        general, cond = aggregate(clf.classes, p)
        status, label, conf = decide(general, clf.min_confidence)
        top = max(general, key=lambda k: general[k])
        subs = select_subtopics(cond.get(top, {}), clf.subtopic_threshold)
        rows.append({"text": text, "argmax": top, "confidence": round(conf, 4),
                     "status": status, "label": label,
                     "thresholded": label if status == STATUS_OK else OTHER_LABEL,
                     "top_subtopic": subs[0][0] if subs else None,
                     "general_scores": {k: round(v, 4) for k, v in general.items()}})
    return rows


def eval_split(clf, frame: pd.DataFrame) -> tuple[dict, list[dict]]:
    rows = batch_outputs(clf, frame["text"].tolist())
    y = frame["general"].tolist()
    argmax = [r["argmax"] for r in rows]
    thr = [r["thresholded"] for r in rows]
    conf = np.array([r["confidence"] for r in rows])
    correct = np.array(argmax) == np.array(y)
    m_arg = classification_metrics(y, argmax, LABELS)
    m_thr = classification_metrics(y, thr, LABELS)
    in_tax = frame["subtopic"].notna().to_numpy()
    path_ok = [(r["argmax"] == g and r["top_subtopic"] == s)
               for r, g, s in zip(rows, frame["general"], frame["subtopic"], strict=True)]
    result = {
        "argmax": summary(m_arg),
        "with_threshold": summary(m_thr),
        "per_class_with_threshold": {c: {k: round(m_thr["per_class"][c][k], 4) for k in
                                         ("precision", "recall", "f1-score", "support")}
                                     for c in LABELS},
        "confusion_matrix_with_threshold": {"labels": LABELS,
                                            "matrix": m_thr["confusion_matrix"]},
        "ece_15bin": round(expected_calibration_error(conf, correct), 4),
        "reliability": reliability_table(conf, correct),
        "subtopic_path_accuracy_in_taxonomy": round(float(np.mean(np.array(path_ok)[in_tax])),
                                                    4) if in_tax.any() else None,
        "uncertain_rate": round(float(np.mean([r["status"] == "uncertain" for r in rows])), 4),
        "out_of_scope_rate": round(float(np.mean([r["status"] == "out_of_scope"
                                                  for r in rows])), 4),
    }
    for r, g, s, t in zip(rows, frame["general"], frame["subtopic"], frame["term"], strict=True):
        r.update({"true_general": g, "true_subtopic": s, "term": t})
    return result, rows


def eval_short_inputs(clf, frame: pd.DataFrame) -> dict:
    """Kısa girdi davranışı: test kartlarının yalnızca terim (1-3 sözcük) alanı."""
    terms = frame["term"].tolist()
    out = {}
    for name, short in (("global_temperature", False), ("short_temperature", True)):
        proba = clf.predict_proba(terms, short=short)
        scores = [aggregate(clf.classes, p)[0] for p in proba]
        pred = [max(s, key=lambda k: s[k]) for s in scores]
        conf = np.array([s[p] for s, p in zip(scores, pred, strict=True)])
        correct = np.array(pred) == frame["general"].to_numpy()
        out[name] = {"accuracy": round(float(correct.mean()), 4),
                     "macro_f1": round(classification_metrics(frame["general"], pred,
                                                              LABELS)["macro_f1"], 4),
                     "mean_confidence": round(float(conf.mean()), 4),
                     "ece_15bin": round(expected_calibration_error(conf, correct), 4)}
    return out


def eval_ood(clf, frame: pd.DataFrame) -> dict:
    rows = batch_outputs(clf, frame["text"].tolist())
    conf = np.array([r["confidence"] for r in rows])
    status = np.array([r["status"] for r in rows])
    per_cat = (pd.DataFrame({"category": frame["category"], "ok": status == STATUS_OK})
               .groupby("category")["ok"].mean().sort_values(ascending=False))
    return {
        "n": len(rows),
        "rejected_rate": round(float(np.mean(status != STATUS_OK)), 4),
        "predicted_other_rate": round(float(np.mean(status == "out_of_scope")), 4),
        "uncertain_rate": round(float(np.mean(status == "uncertain")), 4),
        "false_accept_rate": round(float(np.mean(status == STATUS_OK)), 4),
        "false_accept_mean_confidence": round(float(conf[status == STATUS_OK].mean()), 4)
        if (status == STATUS_OK).any() else None,
        "worst_categories_false_accept": {k: round(float(v), 3) for k, v in
                                          per_cat.head(6).items()},
        "false_accept_topics": pd.Series([r["label"] for r in rows if r["status"] == STATUS_OK])
        .value_counts().to_dict(),
    }


def eval_external(clf, frame: pd.DataFrame) -> dict:
    out = {}
    for source, grp in frame.groupby("source"):
        rows = batch_outputs(clf, grp["text"].tolist())
        target = grp["general"].iloc[0]
        argmax = np.array([r["argmax"] for r in rows])
        status = np.array([r["status"] for r in rows])
        labels = np.array([r["label"] for r in rows])
        out[source] = {
            "n": len(rows), "target": target,
            "recall_argmax": round(float(np.mean(argmax == target)), 4),
            "recall_with_threshold": round(float(np.mean(labels == target)), 4),
            "uncertain_rate": round(float(np.mean(status == "uncertain")), 4),
            "out_of_scope_rate": round(float(np.mean(status == "out_of_scope")), 4),
            "argmax_distribution": pd.Series(argmax).value_counts().to_dict(),
        }
    return out


def eval_named(clf, cases) -> list[dict]:
    results = []
    for text, general, sub in cases:
        pred = clf.predict(text)
        actual_sub = pred.subtopics[0][0] if pred.subtopics else None
        ok = pred.general == general and (sub is None or actual_sub == sub)
        results.append({"text": text, "expected": f"{general}" + (f" > {sub}" if sub else ""),
                        "actual": f"{pred.general}" + (f" > {actual_sub}" if actual_sub else ""),
                        "status": pred.status, "confidence": pred.confidence,
                        "top_scores": dict(list(pred.general_scores.items())[:3]),
                        "pass": bool(ok)})
    return results


def peak_rss_mb() -> float | None:
    """Tepe bellek (MB). `resource` yalnızca Unix'te vardır; Windows'ta None döner."""
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def performance(clf, texts: list[str], load_seconds: float) -> dict:
    t0 = time.perf_counter()
    for text in texts[:500]:
        normalize(text)
    prep_ms = (time.perf_counter() - t0) / min(500, len(texts)) * 1000
    lat = []
    for text in texts[:300]:
        t = time.perf_counter()
        clf.predict(text)
        lat.append((time.perf_counter() - t) * 1000)
    t0 = time.perf_counter()
    clf.predict_proba_auto(texts[:1000])
    batch_ms = (time.perf_counter() - t0) / min(1000, len(texts)) * 1000
    return {
        "model_load_seconds": round(load_seconds, 3),
        "preprocessing_ms_per_text": round(prep_ms, 4),
        "inference_latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "inference_latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "batch_inference_ms_per_text": round(batch_ms, 4),
        "peak_rss_mb": peak_rss_mb(),
        "artifact_size_mb": round(MODEL_PATH.stat().st_size / 1e6, 2),
    }


def plot_confusion(cm: list[list[int]], labels: list[str], path, title: str) -> None:
    cm_arr = np.array(cm, dtype=float)
    norm = cm_arr / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm_arr[i, j]:
                ax.text(j, i, str(int(cm_arr[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(tables: dict[str, list[dict]], path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="mükemmel kalibrasyon")
    for (name, table), color in zip(tables.items(), (BLUE, ORANGE), strict=False):
        ax.plot([r["mean_confidence"] for r in table], [r["accuracy"] for r in table],
                marker="o", markersize=6, linewidth=2, color=color, label=name)
    ax.set_xlabel("Ortalama güven")
    ax.set_ylabel("Doğruluk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#e6e8eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Reliability diagram (genel konu)", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    setup_logging("WARNING")
    try:
        t0 = time.perf_counter()
        clf = TopicClassifier.load(MODEL_PATH)
        load_s = time.perf_counter() - t0
        splits = {n: load_split(n) for n in ("val", "test", "ood_unseen", "external")}
    except (ModelArtifactError, DatasetError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    test_result, test_rows = eval_split(clf, splits["test"])
    report: dict[str, Any] = {
        "model_version": clf.version,
        "thresholds": {"min_confidence": clf.min_confidence,
                       "subtopic_threshold": clf.subtopic_threshold},
        "test": test_result,
        "short_inputs_test_terms": eval_short_inputs(clf, splits["test"]),
        "ood_unseen": eval_ood(clf, splits["ood_unseen"]),
        "external": eval_external(clf, splits["external"]),
        "quantum_tests": eval_named(clf, QUANTUM_TESTS),
        "acceptance_tests": eval_named(clf, ACCEPTANCE_TESTS),
        "performance": performance(clf, splits["test"]["text"].tolist(), load_s),
        "note": "val satırları final modelin eğitim verisine dahildir; val burada yalnızca "
                "reliability karşılaştırması için gösterilmez. Tüm test/OOD/harici metrikleri "
                "eğitimde görülmemiş veridir.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (REPORTS_DIR / "predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(test_result["confusion_matrix_with_threshold"]["matrix"], LABELS,
                   FIGURES_DIR / "confusion_matrix_test.png",
                   "Test confusion matrix (satır-normalize renk, hücrede adet)")
    training_report = REPORTS_DIR / "training_report.json"
    tables = {"test (temperature)": test_result["reliability"]}
    if training_report.exists():
        tr = json.loads(training_report.read_text(encoding="utf-8"))
        if "isotonic" in tr.get("calibration", {}):
            tables["val (isotonic, reddedildi)"] = tr["calibration"]["isotonic"]["reliability"]
    plot_reliability(tables, FIGURES_DIR / "reliability_diagram.png")

    clf.metadata["metrics"]["test_argmax"] = test_result["argmax"]
    clf.metadata["metrics"]["test_with_threshold"] = test_result["with_threshold"]
    clf.metadata["metrics"]["test_ece"] = test_result["ece_15bin"]
    clf.metadata["metrics"]["ood_rejected_rate"] = report["ood_unseen"]["rejected_rate"]
    write_metadata(clf.metadata, MODEL_PATH.with_suffix(".json"))

    print(json.dumps({"test_with_threshold": test_result["with_threshold"],
                      "test_argmax": test_result["argmax"],
                      "test_ece": test_result["ece_15bin"],
                      "short_inputs": report["short_inputs_test_terms"],
                      "subtopic_path_accuracy": test_result["subtopic_path_accuracy_in_taxonomy"],
                      "ood": {k: report["ood_unseen"][k] for k in
                              ("rejected_rate", "false_accept_rate")},
                      "external": {k: {kk: v[kk] for kk in ("recall_argmax",
                                                             "recall_with_threshold")}
                                   for k, v in report["external"].items()},
                      "quantum": [(t["actual"], t["pass"]) for t in report["quantum_tests"]],
                      "acceptance": [(t["actual"], t["pass"]) for t in
                                     report["acceptance_tests"]],
                      "performance": report["performance"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
