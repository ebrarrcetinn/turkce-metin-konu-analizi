"""Keşifsel veri analizi: istatistikleri reports/eda.json'a, grafikleri reports/figures/'a yazar."""

import json
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, OTHER_LABEL, REPORTS_DIR  # noqa: E402
from src.data.build import load_split, load_taboo_cards  # noqa: E402
from src.preprocessing.text import (  # noqa: E402
    HTML_TAG_RE,
    MENTION_RE,
    URL_RE,
    content_tokens,
    tokenize,
)

EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")
MOJIBAKE_RE = re.compile("Ã|Ä±|ÅŸ|Ã¼|Ã§|Ã¶")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
UNUSUAL_CHAR_RE = re.compile(r"[^\sa-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9.,;:!?()'’\"\-–/%&+]")
# Kesik açıklama sezgisi: son sözcük belirtme/ilgi eki ile bitip ardından isim gelmiyor
TRUNCATION_RE = re.compile(r"(?:ını|ini|unu|ünü|nın|nin|nun|nün|yla|yle|arak|erek)\.?$")
COLOR = "#2f6db3"


def anomaly_counts(texts: pd.Series) -> dict[str, int]:
    return {
        "html": int(texts.str.contains(HTML_TAG_RE).sum()),
        "url": int(texts.str.contains(URL_RE).sum()),
        "emoji": int(texts.str.contains(EMOJI_RE).sum()),
        "mention": int(texts.str.contains(MENTION_RE).sum()),
        "hashtag": int(texts.str.contains(HASHTAG_RE).sum()),
        "mojibake": int(texts.str.contains(MOJIBAKE_RE).sum()),
        "unusual_chars": int(texts.str.contains(UNUSUAL_CHAR_RE).sum()),
        "likely_truncated_definition": int(texts.str.strip().str.contains(TRUNCATION_RE).sum()),
    }


def _bar(series: pd.Series, title: str, path, horizontal: bool = False, figsize=(8, 4)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    (series.sort_values().plot.barh if horizontal else series.plot.bar)(ax=ax, color=COLOR)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_taboo_cards()
    splits = {n: load_split(n) for n in ("train", "val", "test", "ood_unseen", "external")}
    labeled = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)

    raw_text = raw["term"] + " " + raw["definition"]
    raw["n_words"] = raw_text.map(lambda t: len(t.split()))
    raw["n_chars"] = raw_text.str.len()
    labeled["n_words"] = labeled["text"].map(lambda t: len(t.split()))
    vocab = Counter(t for text in labeled["text"] for t in tokenize(text))
    content_vocab = Counter(t for text in labeled["text"] for t in content_tokens(text))
    non_other = labeled.loc[labeled.general != OTHER_LABEL, "general"].value_counts()

    eda = {
        "raw": {
            "shape": list(raw.shape),
            "columns": list(raw.columns),
            "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
            "nulls": {c: int(v) for c, v in raw.isna().sum().items()},
            "n_categories": int(raw["category"].nunique()),
            "cards_per_category_min_max": [int(raw["category"].value_counts().min()),
                                           int(raw["category"].value_counts().max())],
            "difficulty_distribution": raw["difficulty"].value_counts().to_dict(),
            "duplicate_rows_term_and_definition": int(
                raw.duplicated(subset=["term", "definition"]).sum()),
            "duplicate_terms_across_categories": int(
                raw.assign(t=raw["term"].str.lower()).duplicated(subset=["t"]).sum()),
            "duplicate_definitions": int(raw.duplicated(subset=["definition"]).sum()),
            "words_per_text": raw["n_words"].describe().round(2).to_dict(),
            "chars_per_text": raw["n_chars"].describe().round(2).to_dict(),
            "anomalies": anomaly_counts(raw["definition"]),
        },
        "labeled_seen": {
            "rows": int(len(labeled)),
            "general_distribution": labeled["general"].value_counts().to_dict(),
            "imbalance_ratio_max_min_excl_other": round(non_other.max() / non_other.min(), 2),
            "other_share": round(float((labeled.general == OTHER_LABEL).mean()), 4),
            "subtopic_distribution": labeled["subtopic"].value_counts().to_dict(),
            "vocabulary_size": len(vocab),
            "vocabulary_size_content": len(content_vocab),
            "hapax_ratio": round(sum(1 for c in vocab.values() if c == 1) / len(vocab), 4),
            "top_tokens": vocab.most_common(25),
            "top_content_tokens": content_vocab.most_common(25),
            "top_content_tokens_per_class": {
                g: Counter(t for text in labeled.loc[labeled.general == g, "text"]
                           for t in content_tokens(text)).most_common(10)
                for g in sorted(labeled["general"].unique())
            },
            "words_per_text_by_class": labeled.groupby("general")["n_words"]
            .mean().round(2).to_dict(),
        },
        "external": {
            "rows": int(len(splits["external"])),
            "by_source": splits["external"]["source"].value_counts().to_dict(),
            "words_per_text": splits["external"]["text"].map(lambda t: len(t.split()))
            .describe().round(2).to_dict(),
            "anomalies": anomaly_counts(splits["external"]["text"]),
        },
    }
    (REPORTS_DIR / "eda.json").write_text(
        json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8")

    _bar(labeled["general"].value_counts(), "Genel konu dağılımı (train+val+test)",
         FIGURES_DIR / "label_distribution.png")
    _bar(labeled["subtopic"].value_counts(), "Alt konu dağılımı",
         FIGURES_DIR / "subtopic_distribution.png", horizontal=True, figsize=(9, 8))
    _bar(pd.Series(dict(content_vocab.most_common(20))), "En sık 20 içerik sözcüğü",
         FIGURES_DIR / "top_tokens.png", horizontal=True, figsize=(8, 5))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(labeled["n_words"], bins=range(3, 16), alpha=0.8, label="tabu (etiketli)",
                 color=COLOR, density=True)
    axes[0].hist(splits["external"]["text"].map(lambda t: len(t.split())), bins=range(3, 42),
                 alpha=0.6, label="harici (insan yazımı)", color="#d9822b", density=True)
    axes[0].set_title("Metin başına kelime sayısı (yoğunluk)")
    axes[0].legend()
    axes[1].hist(raw["n_chars"], bins=30, color=COLOR)
    axes[1].set_title("Metin başına karakter sayısı (ham tabu)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=120)
    plt.close(fig)

    print(json.dumps({"anomalies": eda["raw"]["anomalies"],
                      "imbalance": eda["labeled_seen"]["imbalance_ratio_max_min_excl_other"],
                      "vocab": eda["labeled_seen"]["vocabulary_size"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
