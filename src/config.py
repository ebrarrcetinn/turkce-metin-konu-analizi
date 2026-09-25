"""
Dosya   : src/config.py
Konu    : Proje Ayarları
Açıklama: Yollar, veri bölme oranları, model hiperparametreleri, eşikler, sohbet takibi ve web
          arama ayarları gibi tüm sabitleri tek yerde toplar. Bazı değerler ortam
          değişkenleriyle değiştirilebilir.
Yazar   : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Tekrar üretilebilirlik ---
RANDOM_SEED = 42

# --- Yollar ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model iki dosya olarak saklanır: ağırlıklar (.npz) ve sözlük/ayarlar (.json).
MODEL_PATH = Path(os.environ.get("NLP_MODEL_PATH", MODELS_DIR / "topic_model"))
DB_PATH = Path(os.environ.get("NLP_DB_PATH", PROJECT_ROOT / "nlp_app.db"))

# --- Veri bölme (sınıf bazında stratified, terim bazında gruplu) ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15
# OOD testi için eğitimden tamamen ayrılan "Diğer" kategorilerinin oranı
UNSEEN_OOD_FRACTION = 0.35

# --- Model ---
MODEL_VERSION = "2.0.0"
NB_ALPHA = 0.001  # Naive Bayes Laplace düzeltmesi (200 bin özellikte küçük olmalı)
SOFTMAX_LEARNING_RATE = 0.05
SOFTMAX_L2 = 1e-6
SOFTMAX_MAX_EPOCHS = 30

# --- Çıkarım ---
OTHER_LABEL = "Diğer"
UNCERTAIN_LABEL = "Belirsiz"
# Güven eşiği eğitimde doğrulama verisiyle bu ızgaradan seçilir.
CONFIDENCE_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))  # 0.20 ... 0.90
MAX_SUBTOPICS = 3
# İlk alt konudan sonrakiler, genel konu içindeki payı bu değeri aşarsa listelenir.
SUBTOPIC_MIN_SCORE = 0.15
# Ana konu dışında bu olasılığı aşan genel konular "ilişkili konu" olarak gösterilir.
RELATED_TOPIC_MIN_SCORE = 0.20
MAX_INPUT_CHARS = 5000


def _env_float(name: str, default: float) -> float:
    """Geçersiz ortam değişkeninde çökme yerine varsayılan değer."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --- Sohbet takibi ---
DECAY = _env_float("NLP_DECAY", 0.7)
# Birleşik sohbet konusuna katılmak için gereken minimum pay. decay 0.7 ile iki mesaj
# önce konuşulan bir konunun ağırlığı ~0.49'a iner; üç konulu sohbette payı 0.15-0.20
# aralığına düştüğü için eşik 0.15 seçildi.
TOPIC_MIN_SHARE = 0.15

# --- Web arama ---
WEB_TIMEOUT = _env_float("NLP_WEB_TIMEOUT", 6.0)
WEB_RETRIES = 2
WEB_BACKOFF_SECONDS = 0.8
MAX_RESULTS = 3
CACHE_TTL_HOURS = 24
WIKIPEDIA_API_URL = "https://tr.wikipedia.org/w/api.php"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
USER_AGENT = (
    "TurkceKonuAnalizi/1.0 (egitim projesi; https://github.com/ebrarrcetinn/book-genre-classifier)"
)
ALLOWED_RESULT_HOSTS = ("wikipedia.org", "duckduckgo.com")
WEB_SEARCH_ENABLED = os.environ.get("NLP_WEB_SEARCH", "1") != "0"
# Bağlantı yoksa web araması bu süre boyunca denenmez.
WEB_OFFLINE_COOLDOWN_SECONDS = 120.0
# Sorguya eklenecek anahtar sözcük için minimum model ağırlığı.
KEYWORD_MIN_WEIGHT = 0.30

# --- Loglama ---
LOG_LEVEL = os.environ.get("NLP_LOG_LEVEL", "WARNING")
LOG_DIR = PROJECT_ROOT / "logs"
