"""Genel konu için aday modelleri train/validation üzerinde karşılaştırır.

Alt komutlar: compare, fasttext, cv, tune, hierarchy. Test seti kullanılmaz;
sonuçlar reports/experiments.json dosyasına eklenir.
"""

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.config import OTHER_LABEL, RANDOM_SEED, REPORTS_DIR
from src.data.build import load_split
from src.evaluation.metrics import classification_metrics, summary
from src.models.pipeline import CANDIDATES, build_estimator, get_candidate
from src.models.taxonomy import ALL_GENERAL_LABELS
from src.preprocessing.text import normalize

RESULTS = REPORTS_DIR / "experiments.json"
LABELS = list(ALL_GENERAL_LABELS)


def _load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def _save_results(results: dict) -> None:
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _size_mb(model) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return round(buffer.tell() / 1e6, 2)


def _latency_ms(predict_one, texts: list[str], n: int = 200) -> float:
    sample = texts[:n]
    start = time.perf_counter()
    for text in sample:
        predict_one(text)
    return round((time.perf_counter() - start) / len(sample) * 1000, 3)


def undersample_other(train: pd.DataFrame, max_other: int | None, seed: int = RANDOM_SEED):
    if max_other is None:
        return train
    other = train[train.general == OTHER_LABEL]
    if len(other) <= max_other:
        return train
    return pd.concat([train[train.general != OTHER_LABEL],
                      other.sample(max_other, random_state=seed)]).sample(frac=1,
                                                                          random_state=seed)


def _report(train_true, train_pred, val_true, val_pred, fit_s, latency, size) -> dict:
    val_m = classification_metrics(val_true, val_pred, LABELS)
    train_m = classification_metrics(train_true, train_pred, LABELS)
    return {
        "train": summary(train_m),
        "val": summary(val_m),
        "val_per_class_f1": {c: round(val_m["per_class"][c]["f1-score"], 4) for c in LABELS},
        "fit_seconds": round(fit_s, 2),
        "latency_ms_per_text": latency,
        "size_mb": size,
    }


def evaluate_general(pipe, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    pipe.fit(train["text"].tolist(), train["general"].tolist())
    fit_s = time.perf_counter() - t0
    return _report(train["general"], pipe.predict(train["text"].tolist()),
                   val["general"], pipe.predict(val["text"].tolist()), fit_s,
                   _latency_ms(lambda t: pipe.predict([t]), val["text"].tolist()),
                   _size_mb(pipe))


def cmd_compare(results: dict, only: set[str] | None) -> None:
    train, val = load_split("train"), load_split("val")
    results.setdefault("compare", {})
    for cand in CANDIDATES:
        if only and cand.exp_id not in only:
            continue
        print(f"[{cand.exp_id}] {cand.name} ...", flush=True)
        res = evaluate_general(cand.build(), train, val)
        clf_params = cand.build().named_steps["clf"].get_params()
        res.update({"name": cand.name, "features": cand.features,
                    "hyperparameters": {k: str(v) for k, v in clf_params.items()}})
        results["compare"][cand.exp_id] = res
        print(f"   val macroF1={res['val']['macro_f1']:.4f} acc={res['val']['accuracy']:.4f} "
              f"train macroF1={res['train']['macro_f1']:.4f} fit={res['fit_seconds']}s "
              f"lat={res['latency_ms_per_text']}ms size={res['size_mb']}MB", flush=True)
        _save_results(results)


FASTTEXT_PARAMS = {"epoch": 25, "lr": 0.5, "wordNgrams": 2, "minn": 2, "maxn": 5, "dim": 100,
                   "loss": "softmax", "thread": 1, "seed": RANDOM_SEED, "verbose": 0}


def _ft_label(label: str) -> str:
    return "__label__" + label.replace(" ", "_")


def cmd_fasttext(results: dict) -> None:
    """E12: fastText supervised (alt-sözcük n-gram'lı, sıfırdan eğitilmiş gömmeler)."""
    import fasttext  # yalnızca deney bağımlılığı (requirements-dev.txt)

    train, val = load_split("train"), load_split("val")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.txt"
        path.write_text("\n".join(f"{_ft_label(g)} {normalize(t)}" for t, g in
                                  zip(train.text, train.general, strict=True)), encoding="utf-8")
        t0 = time.perf_counter()
        model = fasttext.train_supervised(str(path), **FASTTEXT_PARAMS)
        fit_s = time.perf_counter() - t0
        model_path = Path(tmp) / "m.bin"
        model.save_model(str(model_path))
        size = round(model_path.stat().st_size / 1e6, 2)

    def predict(texts):
        labels, _ = model.predict([normalize(t) for t in texts])
        return [lab[0].removeprefix("__label__").replace("_", " ") for lab in labels]

    res = _report(train.general, predict(train.text.tolist()), val.general,
                  predict(val.text.tolist()), fit_s,
                  _latency_ms(lambda t: predict([t]), val.text.tolist()), size)
    res.update({"name": "fastText supervised (subword 2-5, bigram)",
                "features": "sıfırdan eğitilen 100d alt-sözcük gömmeleri",
                "hyperparameters": {k: str(v) for k, v in FASTTEXT_PARAMS.items()}})
    results.setdefault("compare", {})["E12"] = res
    _save_results(results)
    print(f"[E12] val macroF1={res['val']['macro_f1']:.4f} acc={res['val']['accuracy']:.4f} "
          f"train macroF1={res['train']['macro_f1']:.4f} fit={fit_s:.1f}s")


def cmd_cv(results: dict, exp_ids: list[str], n_splits: int = 5) -> None:
    """train+val birleşiminde terim-gruplu 5-kat CV; validation farklarının anlamlılığı için."""
    data = pd.concat([load_split("train"), load_split("val")], ignore_index=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(data, data["category"], data["group"]))
    out = {}
    for exp_id in exp_ids:
        scores = []
        for tr_idx, va_idx in folds:
            pipe = get_candidate(exp_id).build()
            pipe.fit(data.text.iloc[tr_idx].tolist(), data.general.iloc[tr_idx].tolist())
            pred = pipe.predict(data.text.iloc[va_idx].tolist())
            scores.append(classification_metrics(data.general.iloc[va_idx], pred,
                                                 LABELS)["macro_f1"])
        out[exp_id] = {"macro_f1_mean": round(float(np.mean(scores)), 4),
                       "macro_f1_std": round(float(np.std(scores)), 4),
                       "folds": [round(s, 4) for s in scores]}
        print(exp_id, out[exp_id], flush=True)
    results["cv"] = out
    _save_results(results)


def cmd_tune(results: dict, exp_id: str) -> None:
    train, val = load_split("train"), load_split("val")
    rows = []
    for cap in (None, 4000):
        sub_train = undersample_other(train, cap)
        for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            res = evaluate_general(build_estimator(exp_id, C=c), sub_train, val)
            rows.append({"exp_id": exp_id, "C": c, "max_other": cap, **res})
            print(f"C={c} max_other={cap}: val macroF1={res['val']['macro_f1']:.4f} "
                  f"train macroF1={res['train']['macro_f1']:.4f}", flush=True)
    results["tune"] = rows
    best = max(rows, key=lambda r: r["val"]["macro_f1"])
    results["tune_best"] = {"exp_id": exp_id, "C": best["C"], "max_other": best["max_other"],
                            "val_macro_f1": best["val"]["macro_f1"]}
    _save_results(results)
    print("BEST", results["tune_best"])


def cmd_hierarchy(results: dict, exp_id: str, c: float) -> None:
    """Hiyerarşik (genel -> alt konu modeli) ile düz (doğrudan 'genel > alt' sınıfı)."""
    train, val = load_split("train"), load_split("val")
    base = build_estimator(exp_id, C=c)
    lab_train = train[train.subtopic.notna()]
    lab_val = val[val.subtopic.notna()]
    flat_y = [f"{g} > {s}" if s else OTHER_LABEL for g, s in
              zip(train.general, train.subtopic, strict=True)]

    t0 = time.perf_counter()
    flat = clone(base).fit(train["text"].tolist(), flat_y)
    flat_fit = time.perf_counter() - t0
    flat_pred = flat.predict(lab_val["text"].tolist())
    flat_gen = [p.split(" > ")[0] for p in flat_pred]
    flat_sub = [p.split(" > ")[1] if " > " in p else None for p in flat_pred]

    t0 = time.perf_counter()
    general = clone(base).fit(train["text"].tolist(), train["general"].tolist())
    sub_models = {g: clone(base).fit(grp["text"].tolist(), grp["subtopic"].tolist())
                  for g, grp in lab_train.groupby("general") if grp.subtopic.nunique() > 1}
    single = {g: grp.subtopic.iloc[0] for g, grp in lab_train.groupby("general")
              if grp.subtopic.nunique() == 1}
    hier_fit = time.perf_counter() - t0
    hier_gen = list(general.predict(lab_val["text"].tolist()))
    hier_sub = [sub_models[g].predict([t])[0] if g in sub_models else single.get(g)
                for t, g in zip(lab_val["text"], hier_gen, strict=True)]

    def score(gen_pred, sub_pred):
        joint = [f"{a}>{b}" for a, b in zip(lab_val.general, lab_val.subtopic, strict=True)]
        joint_pred = [f"{a}>{b}" for a, b in zip(gen_pred, sub_pred, strict=True)]
        jm = classification_metrics(joint, joint_pred, sorted(set(joint)))
        return {"path_accuracy": round(jm["accuracy"], 4),
                "path_macro_f1": round(jm["macro_f1"], 4),
                "general_accuracy_on_labeled": round(
                    float(np.mean(np.array(gen_pred) == lab_val.general.to_numpy())), 4)}

    results["hierarchy"] = {
        "exp_id": exp_id, "C": c, "n_val_in_taxonomy": int(len(lab_val)),
        "flat": {**score(flat_gen, flat_sub), "fit_seconds": round(flat_fit, 2)},
        "hierarchical": {**score(hier_gen, hier_sub), "fit_seconds": round(hier_fit, 2)},
    }
    _save_results(results)
    print(json.dumps(results["hierarchy"], indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["compare", "fasttext", "cv", "tune", "hierarchy"])
    parser.add_argument("--exp", default="E11")
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    results = _load_results()
    if args.command == "compare":
        cmd_compare(results, set(args.only) if args.only else None)
    elif args.command == "fasttext":
        cmd_fasttext(results)
    elif args.command == "cv":
        cmd_cv(results, args.only or ["E07", "E08", "E11"])
    elif args.command == "tune":
        cmd_tune(results, args.exp)
    else:
        cmd_hierarchy(results, args.exp, args.C)
    return 0


if __name__ == "__main__":
    sys.exit(main())
