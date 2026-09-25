"""
Dosya   : src/models/evaluator.py
Konu    : Model Değerlendirme
Açıklama: Modeli test, görülmemiş konu (OOD) ve harici setlerde ölçer; ödevdeki sohbet
          senaryolarını çalıştırır ve sonuçları reports/degerlendirme_raporu.json
          dosyasına yazar.
Yazar   : Ebrar Cemre Çetin
Tarih   : 25.09.2026
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.config import OTHER_LABEL, PROCESSED_DIR, REPORTS_DIR
from src.data.dataset import Record, load_split
from src.ml.metrics import ClassificationReport, expected_calibration_error
from src.models.taxonomy import ALL_GENERAL_LABELS
from src.models.topic_model import STATUS_OK, TopicModel, aggregate, decide
from src.services.app import ChatService

LABELS = list(ALL_GENERAL_LABELS)

# Ödevdeki örnek sohbetler. Kodda bu cümlelere özel hiçbir kural yoktur; burada yalnızca
# sistemin genel davranışını ödevdeki beklentiyle karşılaştırmak için kullanılırlar.
# Beklenen tema ifadeleri ödev metnindeki ifadelerdir.
CONVERSATION_SCENARIOS: list[dict] = [
    {"name": "Senaryo 1: kitaplar → bilim → biyoloji",
     "messages": ["Kitaplar hakkında konuşalım.", "Bilim ile ilgili neler var?",
                  "Biyoloji hakkında ne önerirsin?"],
     "expected_generals": ["Kitaplar", "Bilim", "Biyoloji"],
     "expected_phrases": ["kitaplar", "bilimsel kitaplar",
                          "biyoloji hakkında bilimsel kitaplar"]},
    {"name": "Senaryo 2: kuantum",
     "messages": ["Kuantum bilgisayarlar nasıl çalışır?", "Kuantum mekaniği neyi açıklar?"],
     "expected_generals": ["Teknoloji", "Fizik"],
     "expected_subtopics": ["Kuantum Bilgisayarlar", "Kuantum Mekaniği"]},
]

# Eğitim verisinde bulunmayan, elle yazılmış cümleler. (metin, beklenen genel konu,
# beklenen alt konu; None ise alt konu denetlenmez)
SENTENCE_TESTS = [
    ("Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir.",
     "Fizik", "Kuantum Mekaniği"),
    ("Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir.",
     "Teknoloji", "Kuantum Bilgisayarlar"),
    ("Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı.", "Kitaplar", None),
    ("Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder.", "Bilim", None),
    ("Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir.", "Biyoloji", None),
]


class Evaluator:
    """Kaydedilmiş modeli eğitimde hiç kullanılmamış setlerde değerlendirir.

    test       : eğitimle aynı kaynaktan ama ortak terim içermeyen kartlar
    ood_unseen : eğitimde ve eşik seçiminde hiç görülmemiş konular
    external   : farklı kaynaktan, farklı üslupta cümleler
    """

    def __init__(self, model: TopicModel, processed_dir: Path = PROCESSED_DIR,
                 reports_dir: Path = REPORTS_DIR):
        self.model = model
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir

    def _decisions(self, texts: list[str]) -> list[dict]:
        """Toplu tahmin: her metin için argmax genel konu, eşikli karar ve alt konu.

        Binlerce metin için tek tek `predict` çağırmak yerine olasılıklar tek matris
        işlemiyle hesaplanır.
        """
        rows = []
        probabilities = self.model.predict_proba(texts)
        for row in probabilities:
            general, conditional = aggregate(self.model.classes, row)
            status, label, confidence = decide(general, self.model.min_confidence)
            top = max(general, key=lambda topic: general[topic])
            subtopics = conditional.get(top, {})
            rows.append({"argmax": top, "status": status, "confidence": confidence,
                         "label": label if status == STATUS_OK else OTHER_LABEL,
                         "subtopic": max(subtopics, key=lambda s: subtopics[s])
                         if subtopics else None})
        return rows

    def test_split(self, records: list[Record]) -> dict:
        """Genel konu metrikleri iki şekilde verilir: eşikli (uygulamanın gerçek davranışı,
        "Belirsiz" yanlış sayılır) ve eşiksiz (yalnızca en olası konu)."""
        rows = self._decisions([str(r["text"]) for r in records])
        y_true = [str(r["general"]) for r in records]
        thresholded = ClassificationReport(y_true, [r["label"] for r in rows], LABELS)
        argmax = ClassificationReport(y_true, [r["argmax"] for r in rows], LABELS)
        # Kalibrasyon ölçümü için: her tahminin doğru olup olmadığı ve güveni
        correct = np.array([r["argmax"] == t for r, t in zip(rows, y_true, strict=True)])
        confidence = np.array([r["confidence"] for r in rows])
        # Alt konu doğruluğu: genel konusu doğru bulunan ve alt konusu olan örneklerde.
        pairs = [(r["subtopic"], rec["subtopic"]) for r, rec in zip(rows, records, strict=True)
                 if rec["subtopic"] and r["argmax"] == rec["general"]]
        return {
            "thresholded": thresholded.summary(),
            "argmax": argmax.summary(),
            "per_class": thresholded.per_class(),
            "confusion_matrix": {"labels": LABELS, "matrix": thresholded.confusion.tolist()},
            "subtopic_accuracy": round(sum(p == t for p, t in pairs) / len(pairs), 4)
            if pairs else None,
            "ece_15bin": round(expected_calibration_error(confidence, correct), 4),
            "status_counts": dict(Counter(r["status"] for r in rows)),
        }

    def short_inputs(self, records: list[Record]) -> dict:
        """Yalnızca terim (1-3 sözcük) verildiğinde başarı; açıklama olmadan ne kadar
        tanıyabildiğini gösterir (ör. yalnızca "kübit")."""
        rows = self._decisions([str(r["term"]) for r in records])
        y_true = [str(r["general"]) for r in records]
        report = ClassificationReport(y_true, [r["argmax"] for r in rows], LABELS)
        return {"accuracy": round(report.accuracy, 4), "macro_f1": round(report.macro_f1, 4)}

    def ood(self, records: list[Record]) -> dict:
        """Eğitimde hiç görülmemiş kategoriler: doğru davranış konu iddia etmemektir."""
        rows = self._decisions([str(r["text"]) for r in records])
        status = Counter(r["status"] for r in rows)
        per_category: dict[str, list[bool]] = defaultdict(list)
        for record, row in zip(records, rows, strict=True):
            per_category[str(record["category"])].append(row["status"] == STATUS_OK)
        worst = sorted(((c, float(np.mean(v))) for c, v in per_category.items()),
                       key=lambda item: -item[1])[:6]
        return {"n": len(rows),
                "rejected_rate": round(1 - status[STATUS_OK] / len(rows), 4),
                "status_counts": dict(status),
                "worst_categories_false_accept": {c: round(v, 3) for c, v in worst}}

    def external(self, records: list[Record]) -> dict:
        """Farklı üsluptaki insan yazımı cümleler (biyoloji ders kitabı ve Osmanlı tarihi
        paragrafları). Her kaynağın tek bir hedef konusu vardır; o konuyu bulma oranı ölçülür."""
        out = {}
        for source in sorted({str(r["source"]) for r in records}):
            group = [r for r in records if r["source"] == source]
            rows = self._decisions([str(r["text"]) for r in group])
            target = group[0]["general"]
            out[source] = {"n": len(rows), "target": target,
                           "recall_argmax": round(float(np.mean(
                               [r["argmax"] == target for r in rows])), 4),
                           "recall_with_threshold": round(float(np.mean(
                               [r["label"] == target for r in rows])), 4),
                           "argmax_distribution": dict(Counter(r["argmax"] for r in rows))}
        return out

    def sentences(self) -> list[dict]:
        """Elle yazılmış kabul cümlelerini tek tek dener."""
        results = []
        for text, general, subtopic in SENTENCE_TESTS:
            prediction = self.model.predict(text)
            actual_sub = prediction.subtopics[0][0] if prediction.subtopics else None
            passed = prediction.general == general and subtopic in (None, actual_sub)
            results.append({"text": text, "expected": general + (f" > {subtopic}"
                                                                 if subtopic else ""),
                            "actual": f"{prediction.general}"
                                      + (f" > {actual_sub}" if actual_sub else ""),
                            "confidence": prediction.confidence, "pass": passed})
        return results

    def conversations(self) -> list[dict]:
        """Ödevdeki sohbet senaryolarını web ve veritabanı olmadan çalıştırır."""
        results = []
        for scenario in CONVERSATION_SCENARIOS:
            # Her senaryo yeni bir sohbet olarak başlar; önceki senaryonun konusu taşınmaz.
            service = ChatService(self.model, db=None, web_enabled=False)
            turns: list[dict] = []
            for message in scenario["messages"]:
                turn = service.process(message)
                turns.append({"message": message, "general": turn.prediction.general,
                              "subtopics": [s for s, _ in turn.prediction.subtopics],
                              "theme": turn.theme.label if turn.theme else None,
                              "phrase": turn.theme.phrase if turn.theme else None,
                              "query": turn.query})
            passed = [t["general"] for t in turns] == scenario["expected_generals"]
            if "expected_phrases" in scenario:
                passed &= [t["phrase"] for t in turns] == scenario["expected_phrases"]
            if "expected_subtopics" in scenario:
                passed &= all(expected in t["subtopics"] for t, expected in
                              zip(turns, scenario["expected_subtopics"], strict=True))
            results.append({"name": scenario["name"], "turns": turns, "pass": bool(passed)})
        return results

    def latency(self, texts: list[str]) -> dict:
        """Tek metin tahmin süresi (milisaniye), uygulamadaki gibi tek tek ölçülür."""
        timings = []
        for text in texts:
            start = time.perf_counter()
            self.model.predict(text)
            timings.append((time.perf_counter() - start) * 1000)
        return {"n": len(timings), "median_ms": round(float(np.median(timings)), 3),
                "p95_ms": round(float(np.percentile(timings, 95)), 3)}  # %95'i bundan hızlı

    def run(self) -> dict:
        test = load_split("test", self.processed_dir)
        report = {
            "model": {k: self.model.metadata.get(k) for k in
                      ("model_name", "model_version", "algorithm", "training_timestamp",
                       "thresholds")},
            "test": self.test_split(test),
            "short_inputs": self.short_inputs(test),
            "ood_unseen": self.ood(load_split("ood_unseen", self.processed_dir)),
            "external": self.external(load_split("external", self.processed_dir)),
            "sentence_tests": self.sentences(),
            "conversation_scenarios": self.conversations(),
            "latency": self.latency([str(r["text"]) for r in test[:300]]),
        }
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "degerlendirme_raporu.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
