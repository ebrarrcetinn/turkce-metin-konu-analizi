"""Merkezi yapılandırma. Tüm sabitler burada; bazıları ortam değişkenleriyle ezilebilir."""

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
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = Path(os.environ.get("NLP_MODEL_PATH", MODELS_DIR / "topic_model.joblib"))
MODEL_METADATA_PATH = MODEL_PATH.with_suffix(".json")
DB_PATH = Path(os.environ.get("NLP_DB_PATH", PROJECT_ROOT / "nlp_app.db"))

# --- Veri bölme (sınıf bazında stratified, terim bazında gruplu) ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15
# OOD testi için eğitimden tamamen ayrılan "Diğer" kategorilerinin oranı
UNSEEN_OOD_FRACTION = 0.35

# --- Model ---
FINAL_EXPERIMENT_ID = "E11"
FINAL_C = 1.0
MODEL_VERSION = "1.0.0"

# --- Çıkarım ---
OTHER_LABEL = "Diğer"
UNCERTAIN_LABEL = "Belirsiz"
# Asıl eşikler eğitimde seçilip model metadata'sına yazılır; bunlar yalnızca varsayılan.
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_SUBTOPIC_THRESHOLD = 0.30
THRESHOLD_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(11))  # 0.20 ... 0.70
CONFIDENCE_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))  # 0.20 ... 0.90
MAX_SUBTOPICS = 3
# Bu kadar veya daha az içerik sözcüğü olan girdiler ayrı bir sıcaklıkla kalibre edilir.
SHORT_INPUT_MAX_TOKENS = 3
MAX_INPUT_CHARS = 5000

def _env_float(name: str, default: float) -> float:
    """Geçersiz ortam değişkeninde çökme yerine varsayılan değer."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --- Sohbet takibi ---
DECAY = _env_float("NLP_DECAY", 0.7)
TOPIC_MIN_SHARE = 0.20  # birleşik sohbet konusuna katılmak için gereken minimum pay

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
