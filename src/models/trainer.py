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

GENERAL_LABELS = list(ALL_GENERAL_LABELS)


def joint_labels(rows: list[Record]) -> list[str]:
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

    1. Vektörleştirici eğitim verisine uydurulur (sözlük ve IDF yalnızca train'den).
    2. Naive Bayes ve softmax regresyon doğrulama setinde genel konu macro F1 ile
       karşılaştırılır; iyi olan seçilir.
    3. Seçilen modelin doğrulama logitleriyle sıcaklık (temperature) ve güven eşiği seçilir.
    4. Model train + validation birleşimiyle aynı ayarlarla yeniden eğitilip kaydedilir.
    Test ve OOD setlerine bu aşamada hiç dokunulmaz.
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
        train_texts, val_texts = [str(r["text"]) for r in train], [str(r["text"]) for r in val]
        y_train, y_val = joint_labels(train), joint_labels(val)
        val_general = [str(r["general"]) for r in val]

        vectorizer = build_vectorizer()
        x_train = vectorizer.fit_transform(train_texts)
        x_val = vectorizer.transform(val_texts)
        logger.info("Özellik sayısı: %d", vectorizer.n_features)

        # --- 1) Naive Bayes (temel model) ---
        nb = MultinomialNaiveBayes(alpha=NB_ALPHA).fit(x_train, y_train)
        nb_logits = nb.decision_function(x_val)
        nb_f1 = general_macro_f1(val_general,
                                 general_predictions(nb.classes, softmax(nb_logits)))

        # --- 2) Softmax regresyon (erken durdurma doğrulama macro F1'e göre) ---
        def validation_score(model: SoftmaxRegression) -> float:
            probabilities = model.predict_proba(x_val)
            return general_macro_f1(val_general, general_predictions(model.classes,
                                                                     probabilities))

        softmax_model = SoftmaxRegression(learning_rate=SOFTMAX_LEARNING_RATE, l2=SOFTMAX_L2,
                                          epochs=self.max_epochs, seed=RANDOM_SEED)
        softmax_model.fit(x_train, y_train, validation_score)
        sm_logits = softmax_model.decision_function(x_val)
        sm_f1 = general_macro_f1(val_general, general_predictions(softmax_model.classes,
                                                                  softmax(sm_logits)))
        use_softmax = sm_f1 >= nb_f1
        classes = softmax_model.classes if use_softmax else nb.classes
        logits = sm_logits if use_softmax else nb_logits
        logger.info("Doğrulama macro F1: Naive Bayes %.4f, softmax %.4f", nb_f1, sm_f1)

        # --- 3) Kalibrasyon ve güven eşiği ---
        index = {c: i for i, c in enumerate(classes)}
        targets = np.array([index[label] for label in y_val])
        temperature = fit_temperature(logits, targets)
        # Eşik; doğrulama seti ile eğitimde hiç görülmemiş kategorilerin ayrılmış yarısı
        # (ood_val) birlikte kullanılarak seçilir. Böylece eşik hem doğru konuları kabul
        # etmeyi hem de tanımadığı konularda "Belirsiz" demeyi dengeler.
        ood_val = load_split("ood_val", self.processed_dir)
        ood_logits = (softmax_model if use_softmax else nb).decision_function(
            vectorizer.transform([str(r["text"]) for r in ood_val]))
        probabilities = softmax(np.vstack([logits, ood_logits]) / temperature)
        threshold_truth = val_general + [str(r["general"]) for r in ood_val]
        grid = []
        for threshold in CONFIDENCE_GRID:
            predicted = general_predictions(classes, probabilities, threshold)
            report = ClassificationReport(threshold_truth, predicted, GENERAL_LABELS)
            grid.append({"threshold": threshold, **report.summary()})
        best = max(grid, key=lambda row: (row["macro_f1"], -row["threshold"]))
        min_confidence = float(best["threshold"])

        # --- 4) train + val ile son eğitim ---
        all_texts, all_labels = train_texts + val_texts, y_train + y_val
        final_vectorizer = build_vectorizer()
        x_all = final_vectorizer.fit_transform(all_texts)
        if use_softmax:
            final = SoftmaxRegression(learning_rate=SOFTMAX_LEARNING_RATE, l2=SOFTMAX_L2,
                                      epochs=softmax_model.best_epoch, seed=RANDOM_SEED)
            final.fit(x_all, all_labels)
            weights, bias = final.weights, final.bias
        else:
            final_nb = MultinomialNaiveBayes(alpha=NB_ALPHA).fit(x_all, all_labels)
            weights = final_nb.feature_log_prob.astype(np.float32)
            bias = final_nb.class_log_prior.astype(np.float32)

        algorithm = "softmax_regression" if use_softmax else "multinomial_naive_bayes"
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
            "weights_sha256": hashlib.sha256(weights.tobytes() + bias.tobytes()).hexdigest(),
        }
        model = TopicModel(final_vectorizer, classes, weights, bias, temperature,
                           min_confidence, metadata)
        model.save(self.model_path)

        self.report = {**metadata, "softmax_history": softmax_model.history,
                       "confidence_grid": grid,
                       "training_seconds": round(time.perf_counter() - started, 1)}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "egitim_raporu.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        return model


def _dataset_info(n_train: int, n_val: int, n_ood_val: int) -> dict:
    """Veri kaynağının sürümü (Tabu deposunun commit'i) ve eğitim boyutları."""
    info: dict = {"version": "bilinmiyor", "train": n_train, "val": n_val, "ood_val": n_ood_val}
    path = RAW_DIR / "provenance.json"
    if path.exists():
        provenance = json.loads(path.read_text(encoding="utf-8"))
        info["version"] = provenance.get("taboo", {}).get("commit", "bilinmiyor")[:12]
    return info
