"""
Dosya   : src/config.py
Konu    : Proje Ayarları
Açıklama: Bu dosyada amacım yollar, veri bölme oranları, model hiperparametreleri, eşikler,
          sohbet takibi ve web arama ayarları gibi tüm sabitleri tek yerde toplamak. Bazı
          değerleri ortam değişkenleriyle değiştirilebilir yaptım.
Yazar   : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import os
from pathlib import Path

# Proje kök klasörü: bu dosya src/ içinde olduğu için iki üst klasöre çıkıyorum.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Tekrar üretilebilirlik ---
# Veri bölme, karıştırma ve eğitimdeki tüm rastgele işlemleri bu tohumla yapıyorum; aynı
# veriyle her çalıştırmada aynı modeli ve aynı sonuçları elde ediyorum.
RANDOM_SEED = 42

# --- Yollar ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # GitHub'dan indirdiğim ham veri
PROCESSED_DIR = DATA_DIR / "processed"  # temizleyip böldüğüm veri (JSONL)
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Modeli iki dosya olarak saklıyorum: ağırlıklar (.npz) ve sözlük/ayarlar (.json).
MODEL_PATH = Path(os.environ.get("NLP_MODEL_PATH", MODELS_DIR / "topic_model"))
DB_PATH = Path(os.environ.get("NLP_DB_PATH", PROJECT_ROOT / "nlp_app.db"))

# --- Veri bölme (konu bazında tabakalı, terim bazında gruplu) ---
TEST_SIZE = 0.15  # verinin %15'i test
VAL_SIZE = 0.15  # %15'i doğrulama (model seçimi, kalibrasyon, eşik); kalan %70 eğitim
# "Diğer" kategorilerinin %35'ini eğitimden tamamen ayırıyorum: model hiç görmediği konularda
# "emin değilim" diyebiliyor mu, bunu bununla ölçüyorum.
UNSEEN_OOD_FRACTION = 0.35

# --- Model ---
MODEL_VERSION = "2.0.0"
# Naive Bayes Laplace düzeltmesi. Klasik değer 1; ancak ~200 bin özellikte her birine 1
# eklediğimde gerçek sayımlar bastırıldı ve bütün sınıflar birbirine benzedi. Doğrulama macro F1:
# alpha 1 → 0,08; 0,1 → 0,51; 0,01 → 0,82; 0,001 → 0,83.
NB_ALPHA = 0.001
# Softmax regresyon ayarlarını doğrulama setinde deneyerek seçtim.
SOFTMAX_LEARNING_RATE = 0.05
SOFTMAX_L2 = 1e-6
SOFTMAX_MAX_EPOCHS = 30  # üst sınır; gerçek epoch sayısını erken durdurma belirler

# --- Tahmin ---
OTHER_LABEL = "Diğer"  # taksonomi dışı konular
UNCERTAIN_LABEL = "Belirsiz"  # model yeterince emin değil
# Güven eşiğini eğitimde doğrulama verisiyle bu değerleri deneyerek seçiyorum: 0,20 ... 0,90
CONFIDENCE_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))
MAX_SUBTOPICS = 3
# İlk alt konudan sonrakileri, genel konu içindeki payları bu değeri aşarsa listeliyorum.
SUBTOPIC_MIN_SCORE = 0.15
# Ana konu dışında bu olasılığı aşan genel konuları "ilişkili konu" olarak gösteriyorum.
RELATED_TOPIC_MIN_SCORE = 0.20
MAX_INPUT_CHARS = 5000  # daha uzun metinleri kırpıyorum


def _env_float(name: str, default: float) -> float:
    """Ortam değişkenini sayıya çeviriyorum; geçersizse çökmek yerine varsayılanı döndürüyorum."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --- Sohbet takibi ---
# Her yeni mesajda eski konu skorlarını bu katsayıyla çarpıyorum: yakın mesajlar daha etkili,
# eski mesajların etkisi yavaşça azalıyor (0,7 → 0,49 → 0,34 ...).
DECAY = _env_float("NLP_DECAY", 0.7)
# Birleşik sohbet konusuna katılmak için gereken minimum pay. decay 0.7 ile iki mesaj
# önce konuşulan bir konunun ağırlığı ~0.49'a iner; üç konulu sohbette payı 0.15-0.20
# aralığına düştüğü için eşiği 0.15 seçtim.
TOPIC_MIN_SHARE = 0.15

# --- Web arama ---
WEB_TIMEOUT = _env_float("NLP_WEB_TIMEOUT", 6.0)  # saniye; cevap vermeyen sunucuyu beklemiyorum
WEB_RETRIES = 2  # geçici hatalarda (zaman aşımı, 5xx) tekrar deneme sayısı
WEB_BACKOFF_SECONDS = 0.8  # denemeler arası bekleme (her denemede artar)
MAX_RESULTS = 3  # gösterilen ve kaydedilen sonuç sayısı
CACHE_TTL_HOURS = 24  # aynı sorgu bu süre içinde tekrar sorulursa veritabanından cevaplıyorum
WIKIPEDIA_API_URL = "https://tr.wikipedia.org/w/api.php"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
# Wikipedia API kuralları, isteklerde uygulamayı tanıtan bir User-Agent ister.
USER_AGENT = (
    "TurkceKonuAnalizi/1.0 (egitim projesi; https://github.com/ebrarrcetinn/book-genre-classifier)"
)
# Yalnızca bu alan adlarından gelen bağlantıları gösteriyor ve kaydediyorum.
ALLOWED_RESULT_HOSTS = ("wikipedia.org", "duckduckgo.com")
WEB_SEARCH_ENABLED = os.environ.get("NLP_WEB_SEARCH", "1") != "0"
# Bağlantı yoksa web aramasını bu süre boyunca denemiyorum; her mesajda zaman aşımı
# beklemiyorum.
WEB_OFFLINE_COOLDOWN_SECONDS = 120.0
# Sorguya ekleyeceğim anahtar sözcük için minimum model ağırlığı.
KEYWORD_MIN_WEIGHT = 0.30

# --- Loglama ---
LOG_LEVEL = os.environ.get("NLP_LOG_LEVEL", "WARNING")
LOG_DIR = PROJECT_ROOT / "logs"
