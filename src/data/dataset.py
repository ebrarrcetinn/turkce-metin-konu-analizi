"""
Dosya   : src/data/dataset.py
Konu    : Veri Seti Hazırlama
Açıklama: Ham Tabu kartlarını okur, taksonomiye eşler, tekrar ve çelişkileri temizler;
          eğitim, doğrulama, test, görülmemiş konu (OOD) ve harici değerlendirme setlerini
          oluşturup sızıntı kontrolüyle birlikte data/processed klasörüne yazar.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import (
    OTHER_LABEL,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TEST_SIZE,
    UNSEEN_OOD_FRACTION,
    VAL_SIZE,
)
from src.ml.splitting import grouped_stratified_split
from src.ml.vectorizer import TfidfVectorizer
from src.models.taxonomy import GENERAL_TOPICS, map_card, subtopics_of
from src.preprocessing.text import normalize

logger = logging.getLogger(__name__)

EXPECTED_TABOO_CARDS = 37_278  # veri setinin sabitlenen sürümündeki kart sayısı
# Tabu JSON dosyalarındaki her kartta bulunması gereken alanlar. Diğer alanlar (id,
# yasakli_kelimeler, zorluk) kullanılmaz; eğitim metni kavram + açıklamadır.
REQUIRED_CARD_KEYS = ("kategori", "kelime", "aciklama")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # nokta/soru/ünlem sonrası boşluktan böl
# Harici setteki cümleler, eğitim verisindeki kart uzunluğuna yakın tutulur.
EXTERNAL_MIN_WORDS, EXTERNAL_MAX_WORDS = 6, 40
EXTERNAL_SAMPLE_PER_SOURCE = 400
NEAR_DUPLICATE_COSINE = 0.9  # bu benzerliğin üstündeki iki metin "neredeyse aynı" sayılır

# Bir veri satırı. Anahtarlar: text, general, subtopic, category, term, group, source
Record = dict[str, str | None]


class DatasetError(RuntimeError):
    """Veri bulunamadı veya beklenen biçimde değil."""


def term_group(term: str) -> str:
    """Bölmede kullanılan grup anahtarı. Her sözcüğün ilk 6 harfi alınır; böylece
    "fotosentez" ile "fotosentezin" aynı gruba düşer ve farklı bölmelere sızmaz."""
    return " ".join(token[:6] for token in normalize(term).split())


class DatasetBuilder:
    """Ham kaynaklardan eğitim, doğrulama, test, OOD ve harici değerlendirme setlerini üretir."""

    def __init__(self, raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
                 seed: int = RANDOM_SEED):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.seed = seed
        self.stats: dict = {}

    def load_cards(self) -> list[Record]:
        """data/raw/taboo/data/*.json dosyalarındaki tüm kartları okur ve alanlarını denetler."""
        data_dir = self.raw_dir / "taboo" / "data"
        paths = sorted(data_dir.glob("*.json"))
        if not paths:
            raise DatasetError(f"Tabu verisi bulunamadı: {data_dir}")
        cards: list[Record] = []
        for path in paths:  # her dosya bir kategori (ör. genetik.json)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetError(f"Bozuk JSON dosyası: {path.name}: {exc}") from exc
            for card in payload if isinstance(payload, list) else []:
                missing = [key for key in REQUIRED_CARD_KEYS if key not in card]
                if missing:
                    raise DatasetError(f"{path.name}: eksik alan(lar) {missing}")
                cards.append({"category": str(card["kategori"]).strip(),
                              "term": str(card["kelime"]).strip(),
                              "definition": str(card["aciklama"]).strip()})
        if len(cards) != EXPECTED_TABOO_CARDS:
            logger.warning("Beklenen kart sayısı %d, bulunan %d", EXPECTED_TABOO_CARDS,
                           len(cards))
        return cards

    def clean(self, cards: list[Record]) -> list[Record]:
        """Kartları taksonomiye eşler, tekrarları ve çelişkili etiketleri atar."""
        stats = {"raw_rows": len(cards)}
        records: list[Record] = []
        excluded = 0
        for card in cards:
            if not card["definition"]:  # açıklaması boş kart öğretici değildir
                continue
            mapped = map_card(str(card["category"]), str(card["term"]))
            if mapped is None:
                excluded += 1
                continue
            # Eğitim metni: kavram + açıklaması. Örnek: "kübit kuantum bilgisayarların temel
            # bilgi birimi olan, 0 ve 1 durumlarını aynı anda taşıyabilen ..."
            text = f"{card['term']} {card['definition']}"
            records.append({"text": text, "general": mapped[0], "subtopic": mapped[1],
                            "category": card["category"], "term": card["term"],
                            "group": term_group(str(card["term"])), "source": "taboo"})
        stats["excluded_ambiguous_rows"] = excluded

        # Aynı metin aynı etiketle tekrar ediyorsa bir kez tutulur; farklı etiketle
        # tekrar ediyorsa etiket çelişkisidir ve hepsi atılır.
        labels_by_text: dict[str, set] = {}
        for record in records:
            labels_by_text.setdefault(normalize(str(record["text"])), set()).add(
                (record["general"], record["subtopic"]))
        seen: set[str] = set()
        unique: list[Record] = []
        for record in records:
            key = normalize(str(record["text"]))
            if len(labels_by_text[key]) > 1 or key in seen:
                continue
            seen.add(key)
            unique.append(record)
        stats["duplicates_or_conflicts_removed"] = len(records) - len(unique)
        stats["rows_after_cleaning"] = len(unique)
        self.stats["cleaning"] = stats
        return unique

    def split(self, records: list[Record]) -> dict[str, list[Record]]:
        """Görülmemiş OOD kategorilerini ayırır, kalanı terim gruplu train/val/test'e böler."""
        rng = random.Random(self.seed)
        # "Diğer" kategorilerinin bir kısmı rastgele seçilip eğitimden tamamen çıkarılır.
        # Model bu konuları (ör. balıkçılık, kaligrafi) hiç görmez; testte bunlara konu
        # iddia etmemesi beklenir.
        other_categories = sorted({str(r["category"]) for r in records
                                   if r["general"] == OTHER_LABEL})
        unseen: set[str] = set(rng.sample(other_categories,
                                          round(len(other_categories) * UNSEEN_OOD_FRACTION)))
        seen = [r for r in records if r["category"] not in unseen]
        seen_groups = {r["group"] for r in seen}
        # Eğitimde bulunabilecek bir terimle aynı gruptaki OOD kartları atılır (sızıntı).
        ood = [r for r in records if r["category"] in unseen and r["group"] not in seen_groups]

        def holdout(rows: list[Record], fraction: float, seed: int):
            # Tabakalama kategori üzerinden yapılır: her kategoriden orantılı örnek ayrılır.
            kept, held = grouped_stratified_split([str(r["category"]) for r in rows],
                                                  [str(r["group"]) for r in rows],
                                                  fraction, seed)
            return [rows[i] for i in kept], [rows[i] for i in held]

        # Önce %15 test ayrılır, sonra kalan %85'ten tüm verinin %15'i kadar doğrulama.
        train_val, test = holdout(seen, TEST_SIZE, self.seed)
        train, val = holdout(train_val, VAL_SIZE / (1 - TEST_SIZE), self.seed + 1)
        # Görülmemiş kategorilerin yarısı güven eşiğini seçmek için (ood_val), diğer yarısı
        # yalnızca son değerlendirme için (ood_unseen) ayrılır; ikisi ortak kategori içermez.
        ood_val_categories = set(rng.sample(sorted(unseen), len(unseen) // 2))
        ood_val = [r for r in ood if r["category"] in ood_val_categories]
        ood_test = [r for r in ood if r["category"] not in ood_val_categories]
        self.stats["unseen_ood_categories"] = {
            "ood_val": sorted(str(c) for c in ood_val_categories),
            "ood_unseen": sorted(str(c) for c in unseen - ood_val_categories)}
        return {"train": train, "val": val, "test": test, "ood_val": ood_val,
                "ood_unseen": ood_test}

    def external(self) -> list[Record]:
        """İnsan yazımı, farklı üsluptaki iki kaynaktan tek konulu cümle örnekleri.

        Tabu kartları kısa tanımlardır. Modelin ders kitabı / ansiklopedi üslubundaki
        metinlerde nasıl davrandığını görmek için biyoloji ve Osmanlı tarihi paragraflarından
        cümleler alınır. Bu set eğitimde hiç kullanılmaz.
        """
        specs = [("bquad", sorted((self.raw_dir / "bquad").glob("*.json")), "Biyoloji", None),
                 ("ottoman", sorted((self.raw_dir / "ottoman").rglob("*.json")), "Tarih",
                  "Osmanlı Tarihi")]
        rng = random.Random(self.seed)
        rows: list[Record] = []
        for source, paths, general, subtopic in specs:
            if not paths:
                logger.warning("Harici kaynak bulunamadı: %s", source)
                continue
            # İki kaynak da SQuAD biçimindedir: data -> paragraphs -> context (paragraf metni).
            contexts = [paragraph["context"] for path in paths
                        for document in json.loads(path.read_text(encoding="utf-8"))["data"]
                        for paragraph in document["paragraphs"]]
            # Paragraflar cümlelere bölünür; tekrarlar set ile atılır, sıralama tekrar
            # üretilebilirlik içindir.
            sentences = sorted({s.strip() for c in contexts
                                for s in SENTENCE_SPLIT_RE.split(c.replace("\n", " "))
                                if EXTERNAL_MIN_WORDS <= len(s.split()) <= EXTERNAL_MAX_WORDS})
            sample = rng.sample(sentences, min(EXTERNAL_SAMPLE_PER_SOURCE, len(sentences)))
            rows += [{"text": s, "general": general, "subtopic": subtopic, "category": source,
                      "term": "", "group": f"{source}:{i}", "source": source}
                     for i, s in enumerate(sample)]
        return rows

    @staticmethod
    def taxonomy_seeds() -> list[Record]:
        """Konu ve alt konu adlarının kendisinden oluşan kısa eğitim örnekleri.

        Tabu kartlarında kartın kendi kategorisinin adı (ör. "biyoloji") yasaklı sözcük
        olduğundan açıklamalarda neredeyse hiç geçmez; bu yüzden model tek başına "biyoloji"
        girdisini tanıyamaz. Her alt konu için alt konu adı ve genel konu adı eklenir. Genel
        konu adı her alt konuya bir kez yazıldığından olasılık alt konular arasında paylaşılır.
        Bu örnekler yalnızca eğitim setine girer.
        """
        rows: list[Record] = []
        for general in GENERAL_TOPICS:
            for subtopic in subtopics_of(general):
                for text in (subtopic, general, f"{general} {subtopic}"):
                    rows.append({"text": text, "general": general, "subtopic": subtopic,
                                 "category": "taksonomi", "term": text,
                                 "group": f"taksonomi:{general}", "source": "taxonomy"})
        return rows

    def build(self) -> dict[str, list[Record]]:
        """Tüm adımları çalıştırır ve her seti data/processed/<ad>.jsonl dosyasına yazar."""
        splits = self.split(self.clean(self.load_cards()))
        splits["train"] += self.taxonomy_seeds()
        splits["external"] = self.external()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            # JSONL: her satır bir JSON kaydı; büyük dosyalar satır satır okunabilir.
            with (self.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stats["split_sizes"] = {name: len(rows) for name, rows in splits.items()}
        self.stats["label_distribution"] = {name: dict(Counter(r["general"] for r in rows))
                                            for name, rows in splits.items()}
        self.stats["leakage"] = leakage_report(splits)
        (self.out_dir / "split_report.json").write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return splits


def leakage_report(splits: dict[str, list[Record]]) -> dict:
    """Eğitim ile doğrulama/test arasında sızıntı olup olmadığını ölçer.

    Ortak grup ve birebir aynı metin sayısı 0 olmalıdır; yakın kopya oranı da çok düşük
    olmalıdır. Aksi halde test başarısı gerçekte olduğundan yüksek görünür.
    """
    train = splits["train"]
    train_groups = {r["group"] for r in train}
    train_texts = {normalize(str(r["text"])) for r in train}
    report: dict = {}
    for name in ("val", "test"):
        report[f"group_overlap_train_{name}"] = len(train_groups & {r["group"]
                                                                    for r in splits[name]})
        report[f"exact_text_overlap_train_{name}"] = len(
            train_texts & {normalize(str(r["text"])) for r in splits[name]})
    for name in ("val", "test", "external"):
        report[f"near_duplicate_rate_{name}"] = round(near_duplicate_rate(train, splits[name]), 4)
    return report


def near_duplicate_rate(reference: list[Record], query: list[Record],
                        threshold: float = NEAR_DUPLICATE_COSINE) -> float:
    """query örneklerinden, reference içinde karakter n-gram kosinüs benzerliği eşiği aşan
    bir komşusu olanların oranı."""
    if not reference or not query:
        return 0.0
    vectorizer = TfidfVectorizer("char", (3, 5), preprocessor=normalize)
    ref = vectorizer.fit_transform([str(r["text"]) for r in reference])
    qry = vectorizer.transform([str(r["text"]) for r in query])
    hits = 0
    # Vektörler L2 normlu olduğu için iki vektörün çarpımı doğrudan kosinüs benzerliğidir.
    # Bellek için sorgular 1000'lik parçalar halinde karşılaştırılır.
    for start in range(0, qry.shape[0], 1000):
        similarity = qry[start:start + 1000] @ ref.T
        hits += int((np.asarray(similarity.max(axis=1).todense()).ravel() >= threshold).sum())
    return hits / qry.shape[0]


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> list[Record]:
    """Hazırlanmış bir seti (train, val, test, ood_val, ood_unseen, external) okur."""
    path = processed_dir / f"{name}.jsonl"
    if not path.exists():
        raise DatasetError(f"{path} bulunamadı; önce veri seti hazırlanmalı.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
