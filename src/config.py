"""
Dosya   : src/config.py
Konu    : Proje Ayarları
Açıklama: Projenin bütün ayarlarını (klasör yolları, verinin nasıl bölüneceği, modelin ayarları,
          eşik değerleri, sohbet takibi ve web arama ayarları) bu tek dosyada topladım. Böylece
          bir ayarı değiştirmek istediğimde kodun içinde aramam gerekmiyor, buradan değiştiriyorum.
          Bazı ayarları kodu hiç açmadan, ortam değişkeniyle (bilgisayara verilen dışarıdan ayar)
          de değiştirebiliyorum.
İsim    : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import os
from pathlib import Path

# Projenin ana klasörü. Bu dosya src/ içinde, iki kez yukarı çıkınca ana klasöre ulaşıyorum.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Her seferinde aynı sonuç
# Bilgisayarın "rastgele" seçimleri aslında bir başlangıç sayısına bağlı. Bu sayıyı sabit
# tuttuğum için programı kaç kez çalıştırırsam çalıştırayım aynı modeli ve aynı sonuçları alıyorum.
RANDOM_SEED = 42

# Klasör yolları
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # GitHub'dan indirdiğim, hiç dokunulmamış veri
PROCESSED_DIR = DATA_DIR / "processed"  # temizlediğim ve parçalara ayırdığım veri (JSONL)
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model iki dosyada duruyor: öğrendiği sayılar (.npz) ve kelime listesi ile ayarları (.json).
# Ortam değişkeni verilmişse o yolu, verilmemişse varsayılan yolu kullanıyorum.
MODEL_PATH = Path(os.environ.get("NLP_MODEL_PATH", MODELS_DIR / "topic_model"))
DB_PATH = Path(os.environ.get("NLP_DB_PATH", PROJECT_ROOT / "nlp_app.db"))

# Veriyi bölme
# Veriyi üç parçaya ayırıyorum: eğitim (model öğreniyor), doğrulama (ayarları seçiyorum),
# test (en sonda modeli hiç görmediği veriyle sınıyorum). Her kategoriden her parçaya aynı
# oranda örnek ayırıyorum ve aynı terimi iki farklı parçaya koymuyorum; yoksa model sınavdaki
# soruyu önceden görmüş olurdu.
TEST_SIZE = 0.15  # verinin %15'i test
VAL_SIZE = 0.15  # %15'i doğrulama; geri kalan %70 eğitim
# "Diğer" kategorilerinin %35'ini modele hiç göstermiyorum. Böylece model tanımadığı bir konuyla
# karşılaşınca yanlış tahmin yapmak yerine "emin değilim" diyebiliyor mu, bunu ölçüyorum.
UNSEEN_OOD_FRACTION = 0.35

# Model
MODEL_VERSION = "2.0.0"
# Naive Bayes, eğitimde hiç görmediği bir kelimeye sıfır olasılık vermesin diye her kelimenin
# sayısına küçük bir değer (alpha) ekliyorum. Genelde 1 eklenir; ama yaklaşık 200 bin özellikte
# (kelime ve harf parçası) 1 eklemek gerçek sayıları ezdi ve bütün konular birbirine benzedi.
# Doğrulama verisindeki başarı (macro F1): alpha 1 → 0,08; 0,1 → 0,51; 0,01 → 0,82; 0,001 → 0,83.
NB_ALPHA = 0.001
# Softmax regresyon ayarları. Birkaç değer deneyip doğrulama verisinde en iyi olanı seçtim.
SOFTMAX_LEARNING_RATE = 0.05  # her adımda ağırlıkları ne kadar düzelteceğim
SOFTMAX_L2 = 1e-6  # modelin veriyi ezberlemesini engellemek için verdiğim küçük ceza
SOFTMAX_MAX_EPOCHS = 30  # eğitim verisinin üzerinden en fazla kaç tur geçileceği; model
                         # gelişmeyi bırakırsa daha erken duruyorum

# Tahmin
OTHER_LABEL = "Diğer"  # listemdeki konuların hiçbirine uymayan metinler
UNCERTAIN_LABEL = "Belirsiz"  # model yeterince emin değilse bu etiketi gösteriyorum
# Model ne kadar eminse cevap versin? Bu sınırı 0,20, 0,25, ..., 0,90 değerlerini tek tek
# deneyip doğrulama verisinde en iyi sonucu vereni seçerek belirliyorum.
CONFIDENCE_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))
MAX_SUBTOPICS = 3  # en fazla bu kadar alt konu gösteriyorum
# İlk alt konuyu her zaman gösteriyorum; sonrakileri ancak ana konu içindeki payları bu
# değerden büyükse gösteriyorum.
SUBTOPIC_MIN_SCORE = 0.15
# Ana konu dışında olasılığı bu değeri geçen konuları "ilişkili konu" olarak gösteriyorum.
RELATED_TOPIC_MIN_SCORE = 0.20
MAX_INPUT_CHARS = 5000  # bundan uzun metinlerin fazlasını kesiyorum


def _env_float(name: str, default: float) -> float:
    """Ortam değişkenini okuyup sayıya çeviriyorum. Değer sayı değilse program çökmesin diye
    varsayılan değeri döndürüyorum."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Sohbet takibi
# Her yeni mesaj geldiğinde eski konuların puanını bu sayıyla çarpıp küçültüyorum. Böylece
# son mesajlar daha önemli sayılıyor, eski mesajların etkisi yavaş yavaş azalıyor
# (0,7 → 0,49 → 0,34 ...).
DECAY = _env_float("NLP_DECAY", 0.7)
# Bir konunun "sohbetin konusu" sayılması için sohbetteki payının en az bu kadar olması gerekiyor.
# İki mesaj önce konuşulan konu 0,49 ağırlığa iniyor; üç konunun konuşulduğu bir sohbette
# böyle bir konunun payı 0,15-0,20 arasında kaldığı için sınırı 0,15 seçtim.
TOPIC_MIN_SHARE = 0.15

# Web arama
WEB_TIMEOUT = _env_float("NLP_WEB_TIMEOUT", 6.0)  # saniye; cevap vermeyen siteyi beklemiyorum
WEB_RETRIES = 2  # geçici bir hata olursa (süre doldu, sunucu hatası) kaç kez tekrar deneyeceğim
WEB_BACKOFF_SECONDS = 0.8  # tekrar denemeden önce bekleme süresi (her denemede biraz daha uzun)
MAX_RESULTS = 3  # gösterip kaydettiğim en fazla sonuç sayısı
CACHE_TTL_HOURS = 24  # aynı soru bu sürede tekrar gelirse internete gitmeden kayıttan cevaplıyorum
WIKIPEDIA_API_URL = "https://tr.wikipedia.org/w/api.php"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
# Wikipedia, istek gönderen programın kendini tanıtmasını istiyor; bu metinle tanıtıyorum.
USER_AGENT = (
    "TurkceKonuAnalizi/1.0 (egitim projesi; https://github.com/ebrarrcetinn/book-genre-classifier)"
)
# Güvenlik için yalnızca bu sitelerden gelen bağlantıları gösterip kaydediyorum.
ALLOWED_RESULT_HOSTS = ("wikipedia.org", "duckduckgo.com")
# NLP_WEB_SEARCH=0 verilirse web aramayı kapatıyorum, başka her durumda açık tutuyorum.
WEB_SEARCH_ENABLED = os.environ.get("NLP_WEB_SEARCH", "1") != "0"
# İnternet yoksa bu süre boyunca web aramayı hiç denemiyorum; böylece her mesajda boşuna
# beklemiyorum.
WEB_OFFLINE_COOLDOWN_SECONDS = 120.0
# Aramaya bir kelime ekleyebilmem için modelin o kelimeye en az bu kadar önem vermesi gerekiyor.
KEYWORD_MIN_WEIGHT = 0.30

# Kayıt tutma (log)
# Programın çalışırken tuttuğum notların (log) ne kadar ayrıntılı olacağını buradan seçiyorum.
LOG_LEVEL = os.environ.get("NLP_LOG_LEVEL", "WARNING")
LOG_DIR = PROJECT_ROOT / "logs"
