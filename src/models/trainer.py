"""
Dosya   : src/models/trainer.py
Konu    : Model Eğitimi
Açıklama: Naive Bayes ile softmax regresyonu doğrulama setinde karşılaştırır, iyi olanı
          seçer, sıcaklık ve güven eşiğini belirler ve modeli eğitim + doğrulama verisiyle
          yeniden eğitip kaydeder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 25.09.2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.config import (
    CONFIDENCE_GRID,
    MODEL_PATH,
    MODEL_VERSION,
    NB_ALPHA,
    OTHER_LABEL,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    REPORTS_DIR,
    SOFTMAX_L2,
    SOFTMAX_LEARNING_RATE,
    SOFTMAX_MAX_EPOCHS,
)
from src.data.dataset import Record, load_split
from src.ml.calibration import fit_temperature
from src.ml.metrics import ClassificationReport
from src.ml.naive_bayes import MultinomialNaiveBayes
from src.ml.softmax_regression import SoftmaxRegression, softmax
from src.models.taxonomy import ALL_GENERAL_LABELS
from src.models.topic_model import (
    STATUS_OK,
    TopicModel,
    aggregate,
    build_vectorizer,
    decide,
    joint_label,
)

logger = logging.getLogger(__name__)

# Başarı 9 genel konu (8 konu + Diğer) üzerinden ölçülür; alt konu ayrıca raporlanır.
GENERAL_LABELS = list(ALL_GENERAL_LABELS)


def joint_labels(rows: list[Record]) -> list[str]:
    """Kayıtlardan modelin öğrendiği birleşik etiketleri üretir ("Fizik > Optik")."""
    return [joint_label(str(r["general"]), r["subtopic"]) for r in rows]


def general_predictions(classes: list[str], probabilities: np.ndarray,
                        min_confidence: float | None = None) -> list[str]:
    """Her örnek için genel konu tahmini. Eşik verilirse belirsiz kararlar değerlendirmede
    "Diğer" sayılır (sistem o metin için konu iddia etmemiş olur)."""
    labels = []
    for row in probabilities:
        general, _ = aggregate(classes, row)
        if min_confidence is None:
            labels.append(max(general, key=lambda topic: general[topic]))
        else:
            status, label, _ = decide(general, min_confidence)
            labels.append(label if status == STATUS_OK else OTHER_LABEL)
    return labels


def general_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return ClassificationReport(y_true, y_pred, GENERAL_LABELS).macro_f1


class Trainer:
    """Modeli eğitir: algoritma seçimi, kalibrasyon, eşik seçimi ve son eğitim.

    Veri: data/processed altındaki JSONL dosyaları (DatasetBuilder üretir).
      train   : modelin öğrendiği örnekler
      val     : algoritma seçimi, erken durdurma, sıcaklık ve eşik seçimi
      ood_val : eğitimde hiç olmayan konular; yalnızca eşik seçiminde kullanılır
    Test, ood_unseen ve external setlerine eğitim sırasında hiç dokunulmaz; bunlar yalnızca
    `Evaluator` tarafından en sonda bir kez ölçülür.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR, model_path: Path = MODEL_PATH,
                 reports_dir: Path = REPORTS_DIR, max_epochs: int = SOFTMAX_MAX_EPOCHS):
        self.processed_dir = processed_dir
        self.model_path = model_path
        self.reports_dir = reports_dir
        self.max_epochs = max_epochs
        self.report: dict = {}

    def run(self) -> TopicModel:
        started = time.perf_counter()
        train = load_split("train", self.processed_dir)
        val = load_split("val", self.processed_dir)
        ood_val = load_split("ood_val", self.processed_dir)
        train_texts, val_texts = [str(r["text"]) for r in train], [str(r["text"]) for r in val]
        y_train, y_val = joint_labels(train), joint_labels(val)
        val_general = [str(r["general"]) for r in val]

        # 1) Metinler sayıya çevrilir. Sözlük ve IDF yalnızca eğitim verisinden öğrenilir;
        #    doğrulama metinleri bu sözlükle dönüştürülür (doğrulamaya bilgi sızmaz).
        vectorizer = build_vectorizer()
        x_train = vectorizer.fit_transform(train_texts)
        x_val = vectorizer.transform(val_texts)
        logger.info("Özellik sayısı: %d", vectorizer.n_features)

        # 2) Naive Bayes ve softmax regresyon eğitilip doğrulama setinde karşılaştırılır.
        nb = MultinomialNaiveBayes(alpha=NB_ALPHA).fit(x_train, y_train)
        nb_f1 = general_macro_f1(val_general,
                                 general_predictions(nb.classes, softmax(
                                     nb.decision_function(x_val))))
        softmax_model = self._train_softmax(x_train, y_train, x_val, val_general)
        sm_f1 = general_macro_f1(val_general, general_predictions(
            softmax_model.classes, softmax_model.predict_proba(x_val)))
        logger.info("Doğrulama macro F1: Naive Bayes %.4f, softmax %.4f", nb_f1, sm_f1)
        use_softmax = sm_f1 >= nb_f1
        chosen: SoftmaxRegression | MultinomialNaiveBayes = softmax_model if use_softmax else nb

        # 3) Sıcaklık ve güven eşiği seçilen modelin doğrulama çıktılarıyla belirlenir.
        logits = chosen.decision_function(x_val)
        temperature = self._fit_temperature(chosen.classes, logits, y_val)
        ood_logits = chosen.decision_function(
            vectorizer.transform([str(r["text"]) for r in ood_val]))
        grid = self._threshold_grid(chosen.classes, np.vstack([logits, ood_logits]),
                                    temperature,
                                    val_general + [str(r["general"]) for r in ood_val])
        # En yüksek macro F1'i veren eşik; eşitlikte düşük eşik tercih edilir.
        best = max(grid, key=lambda row: (row["macro_f1"], -row["threshold"]))
        min_confidence = float(best["threshold"])

        # 4) Seçimler bittikten sonra model eğitim + doğrulama verisinin tamamıyla yeniden
        #    eğitilir: daha fazla örnek daha iyi model demektir. Epoch sayısı erken
        #    durdurmanın bulduğu değerdir; sıcaklık ve eşik 3. adımdaki değerlerdir.
        final_vectorizer = build_vectorizer()
        x_all = final_vectorizer.fit_transform(train_texts + val_texts)
        weights, bias = self._fit_final(use_softmax, x_all, y_train + y_val,
                                        softmax_model.best_epoch)

        algorithm = "softmax_regression" if use_softmax else "multinomial_naive_bayes"
        # Eğitim bilgileri model dosyasına ve veritabanındaki model_metadata tablosuna yazılır.
        metadata = {
            "model_name": f"turkce-konu-{algorithm}",
            "model_version": MODEL_VERSION,
            "training_timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "algorithm": algorithm,
            "dataset": _dataset_info(len(train), len(val), len(ood_val)),
            "features": vectorizer.n_features,
            "hyperparameters": {"nb_alpha": NB_ALPHA, "learning_rate": SOFTMAX_LEARNING_RATE,
                                "l2": SOFTMAX_L2, "epochs": softmax_model.best_epoch},
            "metrics": {"val_macro_f1_naive_bayes": round(nb_f1, 4),
                        "val_macro_f1_softmax": round(sm_f1, 4),
                        "val_thresholded": best},
            "thresholds": {"min_confidence": min_confidence,
                           "temperature": round(temperature, 4)},
            # Ağırlıkların parmak izi: hangi kaydın hangi modelle yapıldığı ayırt edilebilir.
            "weights_sha256": hashlib.sha256(weights.tobytes() + bias.tobytes()).hexdigest(),
        }
        model = TopicModel(final_vectorizer, chosen.classes, weights, bias, temperature,
                           min_confidence, metadata)
        model.save(self.model_path)

        self.report = {**metadata, "softmax_history": softmax_model.history,
                       "confidence_grid": grid,
                       "training_seconds": round(time.perf_counter() - started, 1)}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "egitim_raporu.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        return model

    def _train_softmax(self, x_train, y_train: list[str], x_val,
                       val_general: list[str]) -> SoftmaxRegression:
        """Softmax regresyonu, her epoch sonunda doğrulama macro F1'i ölçerek eğitir."""
        def validation_score(model: SoftmaxRegression) -> float:
            return general_macro_f1(val_general, general_predictions(
                model.classes, model.predict_proba(x_val)))

        model = SoftmaxRegression(learning_rate=SOFTMAX_LEARNING_RATE, l2=SOFTMAX_L2,
                                  epochs=self.max_epochs, seed=RANDOM_SEED)
        return model.fit(x_train, y_train, validation_score)

    @staticmethod
    def _fit_temperature(classes: list[str], logits: np.ndarray, y_val: list[str]) -> float:
        index = {c: i for i, c in enumerate(classes)}
        targets = np.array([index[label] for label in y_val])
        return fit_temperature(logits, targets)

    @staticmethod
    def _threshold_grid(classes: list[str], logits: np.ndarray, temperature: float,
                        truth: list[str]) -> list[dict]:
        """Her aday güven eşiği için macro F1.

        Doğrulama seti ile eğitimde hiç görülmemiş kategoriler (ood_val) birlikte kullanılır.
        Eşik düşükse model tanımadığı konulara da konu atar, yüksekse doğru konuları da
        "Belirsiz" sayar; ikisini en iyi dengeleyen eşik seçilir.
        """
        probabilities = softmax(logits / temperature)
        grid = []
        for threshold in CONFIDENCE_GRID:
            predicted = general_predictions(classes, probabilities, threshold)
            report = ClassificationReport(truth, predicted, GENERAL_LABELS)
            grid.append({"threshold": threshold, **report.summary()})
        return grid

    @staticmethod
    def _fit_final(use_softmax: bool, features, labels: list[str],
                   epochs: int) -> tuple[np.ndarray, np.ndarray]:
        """Seçilen algoritmayı tüm veriyle eğitip (ağırlık, bias) döndürür."""
        if use_softmax:
            final = SoftmaxRegression(learning_rate=SOFTMAX_LEARNING_RATE, l2=SOFTMAX_L2,
                                      epochs=epochs, seed=RANDOM_SEED).fit(features, labels)
            return final.weights, final.bias
        # Naive Bayes skoru da x · log P(özellik | sınıf) + log P(sınıf) biçiminde doğrusaldır;
        # aynı TopicModel ile ağırlık ve bias olarak saklanabilir.
        final_nb = MultinomialNaiveBayes(alpha=NB_ALPHA).fit(features, labels)
        return (final_nb.feature_log_prob.astype(np.float32),
                final_nb.class_log_prior.astype(np.float32))


def _dataset_info(n_train: int, n_val: int, n_ood_val: int) -> dict:
    """Veri kaynağının sürümü (Tabu deposunun commit'i) ve eğitim boyutları."""
    info: dict = {"version": "bilinmiyor", "train": n_train, "val": n_val, "ood_val": n_ood_val}
    path = RAW_DIR / "provenance.json"  # sources.fetch_all indirme sırasında yazar
    if path.exists():
        provenance = json.loads(path.read_text(encoding="utf-8"))
        info["version"] = provenance.get("taboo", {}).get("commit", "bilinmiyor")[:12]
    return info
