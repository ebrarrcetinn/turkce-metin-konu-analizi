"""evaluate.py çıktılarından hata analizi üretir: reports/error_analysis.json

Önkoşul: python evaluate.py
"""

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src.config import OTHER_LABEL, REPORTS_DIR
from src.data.build import load_split
from src.models.predictor import STATUS_OK, TopicClassifier
from src.preprocessing.text import content_tokens

N_EXAMPLES = 4
OOD_PROBES = [
    "Akşam pizza söyleyeceğim.",
    "Yarın sabah dişçiye gitmem gerekiyor.",
    "Bu kazağın rengi bana hiç yakışmadı.",
    "Hafta sonu anneannemin bahçesinde domates topladık.",
]
AMBIGUOUS_PROBES = [
    "Einstein'ın hayatını anlatan biyografi kitabını bitirdim.",
    "Osmanlı döneminde yazılmış astronomi kitapları incelendi.",
    "Futbolcuların performansı yapay zeka ile analiz ediliyor.",
    "Harika, yine bilgisayar çöktü, tam da teslim günü!",
    "Deney tüpünde tepkime gerçekleşti.",
    "Gen.",
]


def main() -> int:
    path = REPORTS_DIR / "predictions_test.jsonl"
    if not path.exists():
        print("HATA: önce `python evaluate.py` çalıştırın.", file=sys.stderr)
        return 1
    df = pd.read_json(path, lines=True)
    df["n_words"] = df["text"].str.split().str.len()
    df["error"] = df["thresholded"] != df["true_general"]
    errors = df[df["error"]]

    pairs = Counter(zip(errors["true_general"], errors["thresholded"], strict=True))
    top_pairs = []
    for (true, pred), count in pairs.most_common(8):
        subset = errors[(errors.true_general == true) & (errors.thresholded == pred)]
        top_pairs.append({
            "true": true, "predicted": pred, "count": int(count),
            "examples": [{"text": r.text, "confidence": r.confidence, "status": r.status,
                          "argmax": r.argmax}
                         for r in subset.head(N_EXAMPLES).itertuples()],
        })

    df["len_bucket"] = pd.cut(df["n_words"], bins=[0, 8, 10, 12, 100],
                              labels=["<=8", "9-10", "11-12", ">=13"])
    by_len = df.groupby("len_bucket", observed=True)["error"].agg(["mean", "count"])

    train = pd.concat([load_split("train"), load_split("val")])
    vocab = {t for text in train["text"] for t in content_tokens(text)}
    stems = {t[:5] for t in vocab}

    def oov_ratio(text: str, known: set[str], key) -> float:
        tokens = content_tokens(text)
        return float(np.mean([key(tok) not in known for tok in tokens])) if tokens else 0.0

    df["oov_ratio"] = df["text"].map(lambda t: oov_ratio(t, vocab, lambda tok: tok))
    df["oov_ratio_f5"] = df["text"].map(lambda t: oov_ratio(t, stems, lambda tok: tok[:5]))

    def rank_of_true(row) -> int:
        ordered = sorted(row["general_scores"], key=row["general_scores"].get, reverse=True)
        return ordered.index(row["true_general"]) + 1 if row["true_general"] in ordered else 99

    errors = errors.assign(true_rank=errors.apply(rank_of_true, axis=1))

    clf = TopicClassifier.load()
    probes: dict[str, list[dict]] = {}
    for name, texts in (("ood", OOD_PROBES), ("ambiguous_or_multi_topic", AMBIGUOUS_PROBES)):
        probes[name] = []
        for text in texts:
            p = clf.predict(text)
            probes[name].append({"text": text, "status": p.status, "label": p.general,
                                 "confidence": p.confidence,
                                 "top3": dict(list(p.general_scores.items())[:3])})

    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    external_errors = {}
    ext = load_split("external")
    for source, grp in ext.groupby("source"):
        rows = []
        for text in grp["text"].head(60):
            p = clf.predict(text)
            if p.general != grp["general"].iloc[0]:
                rows.append({"text": text, "status": p.status, "label": p.general,
                             "top_guess": p.top_guess, "confidence": p.confidence})
        external_errors[source] = rows[:N_EXAMPLES + 2]

    ood_fa = evaluation["ood_unseen"]
    report = {
        "n_test": int(len(df)),
        "n_errors_with_threshold": int(df["error"].sum()),
        "error_rate": round(float(df["error"].mean()), 4),
        "errors_rejected_as_uncertain": int((errors["status"] == "uncertain").sum()),
        "errors_rejected_as_out_of_scope": int((errors["status"] == "out_of_scope").sum()),
        "errors_confidently_wrong": int((errors["status"] == STATUS_OK).sum()),
        "errors_true_class_rank2": int((errors["true_rank"] == 2).sum()),
        "most_confused_pairs": top_pairs,
        "error_rate_by_word_count": {str(k): {"error_rate": round(float(v["mean"]), 4),
                                              "n": int(v["count"])}
                                     for k, v in by_len.iterrows()},
        "oov": {
            "mean_oov_ratio_errors": round(float(df.loc[df.error, "oov_ratio"].mean()), 4),
            "mean_oov_ratio_correct": round(float(df.loc[~df.error, "oov_ratio"].mean()), 4),
            "mean_oov_ratio_f5_errors": round(float(df.loc[df.error, "oov_ratio_f5"].mean()), 4),
            "mean_oov_ratio_f5_correct": round(float(df.loc[~df.error, "oov_ratio_f5"].mean()),
                                               4),
        },
        "other_class": {
            "in_scope_sent_to_other": int(((df.true_general != OTHER_LABEL)
                                           & (df.thresholded == OTHER_LABEL)).sum()),
            "other_accepted_as_topic": int(((df.true_general == OTHER_LABEL)
                                            & (df.thresholded != OTHER_LABEL)).sum()),
        },
        "ood_unseen_false_accept_topics": ood_fa["false_accept_topics"],
        "ood_unseen_worst_categories": ood_fa["worst_categories_false_accept"],
        "external_error_examples": external_errors,
        "probes": probes,
    }
    (REPORTS_DIR / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_errors_with_threshold", "error_rate",
                                             "errors_rejected_as_uncertain",
                                             "errors_rejected_as_out_of_scope",
                                             "errors_confidently_wrong",
                                             "errors_true_class_rank2",
                                             "error_rate_by_word_count", "oov", "other_class")},
                     ensure_ascii=False, indent=1))
    for pair in top_pairs[:6]:
        print(pair["true"], "->", pair["predicted"], pair["count"], "|",
              pair["examples"][0]["text"])
    for name, rows in probes.items():
        for r in rows:
            print(name, "|", r["text"], "->", r["status"], r["label"], r["confidence"], r["top3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
