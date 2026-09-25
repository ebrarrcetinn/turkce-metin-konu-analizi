# Türkçe Metin Konu Analizi

Kullanıcının yazdığı Türkçe metnin **genel konusunu** ve **alt konularını** bulan, sohbet
boyunca konuşmanın **genel konusunu** takip eden, bu konudan bir **arama sorgusu** üretip
internette (Türkçe Wikipedia, yedek olarak DuckDuckGo) arayan ve tüm sonuçları **SQLite**
veritabanına kaydeden konsol uygulaması.

Sınıflandırma için kullanılan TF-IDF vektörleştirici, Naive Bayes, softmax regresyon,
kalibrasyon, veri bölme ve değerlendirme metrikleri hazır makine öğrenmesi kütüphaneleri
kullanılmadan, yalnızca `numpy` / `scipy` (dizi ve seyrek matris yapıları) ile yazılmıştır.

Yazar: Ebrar Cemre Çetin

## Ödev gereksinimleri ve karşılıkları

| Gereksinim | Karşılığı |
|---|---|
| Metnin genel konusu ve alt konuları | `TopicModel.predict`: genel konu, güven ve en olası alt konular (birden fazla olabilir); ayrıca ilişkili diğer genel konular |
| Sonuçların veritabanına kaydı ve konsola yazılması | `metinler` tablosu, konsolda `[Metin Analizi]` bölümü |
| Döngü halinde metin alma | `ConsoleApp.run` (`çıkış` / `q` ile biter) |
| Sohbetin genel konusu | `ConversationTracker`: `skor = eski · 0,7 + yeni olasılık`; `sohbet_konulari` tablosu |
| İnternette arama ve sonuçların kaydı | `WebSearchClient` (Wikipedia → DuckDuckGo); `arama_sonuclari` tablosu |
| Senaryo 1: kitaplar → bilim → biyoloji | Tema: "kitaplar" → "bilimsel kitaplar" → "biyoloji hakkında bilimsel kitaplar" |
| Senaryo 2: kuantum (Fizik / Teknoloji) | "kuantum" → Teknoloji (%77), ilişkili konu Fizik (%23); "kuantum mekaniği" → Fizik > Kuantum Mekaniği, "kuantum bilgisayarlar" → Teknoloji > Kuantum Bilgisayarlar |
| Tek dosyadan çalıştırma | `python proje.py` (model yoksa veriyi indirir ve kendisi eğitir) |
| Nesne yönelimli yapı | `ConsoleApp`, `ModelPreparer`, `ChatService`, `TopicModel`, `Trainer`, `Evaluator`, `DatasetBuilder`, `TfidfVectorizer`, `SoftmaxRegression`, `MultinomialNaiveBayes`, `ConversationTracker`, `WebSearchClient`, `Database` |
| Algoritmaları hazır kütüphane yerine kendimiz yazmak | `src/ml/` ve `src/preprocessing/text.py` (ayrıntı aşağıda) |
| Her dosyada ad ve tarih başlığı | Tüm kaynak ve test dosyalarının başında |

## Kurulum

Python 3.11 veya üzeri gerekir.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Bağımlılıklar yalnızca `numpy`, `scipy` ve `requests`'tir.

## Çalıştırma

```bash
python proje.py
```

İlk çalıştırmada model dosyası olmadığı için program sırasıyla veriyi indirir ve SHA-256 ile
doğrular (~17 MB), veri setini hazırlar ve modeli eğitir (toplam birkaç dakika). Sonraki
çalıştırmalarda kayıtlı model doğrudan yüklenir.

| Seçenek | Etkisi |
|---|---|
| `--no-web` | İnternet aramasını kapatır |
| `--egit` | Veriyi yeniden indirip modeli baştan eğitir |
| `--degerlendir` | Modeli test setlerinde ölçer, `reports/degerlendirme_raporu.json` yazar ve çıkar |
| `--db YOL` | Veritabanı dosyası (varsayılan `nlp_app.db`) |
| `--debug` | Ayrıntılı log |

Konsol komutları: `geçmiş` (bu oturumdaki mesajlar), `sıfırla` (yeni sohbet; kayıtlar
silinmez), `yardım`, `çıkış` / `q`.

İsteğe bağlı ortam değişkenleri: `NLP_WEB_SEARCH=0`, `NLP_DB_PATH`, `NLP_MODEL_PATH`,
`NLP_DECAY`, `NLP_WEB_TIMEOUT`, `NLP_LOG_LEVEL`.

## Örnek çalışma (ödevdeki 1. senaryo, `--no-web`)

```
Metin girin:
Kitaplar hakkında konuşalım.

[Metin Analizi]
Genel Konu: Kitaplar
Güven: %79,2
Alt Konular:
  - Çizgi Roman (%49,1)
  - Edebiyat, Roman ve Şiir (%25,9)
  - Masallar (%17,1)

[Sohbet]
Genel sohbet konusu: Kitaplar
Tema: kitaplar
Arama sorgusu: "kitaplar"

Metin girin:
Bilim ile ilgili neler var?

[Metin Analizi]
Genel Konu: Bilim
Güven: %91,5
Alt Konular:
  - Bilim Felsefesi ve Yöntem (%100,0)

[Sohbet]
Genel sohbet konusu: Bilim + Kitaplar
Tema: bilimsel kitaplar
Arama sorgusu: "bilimsel kitaplar"

Metin girin:
Biyoloji hakkında ne önerirsin?

[Metin Analizi]
Genel Konu: Biyoloji
Güven: %60,4
Alt Konular:
  - Genetik (%22,6)
  - Ekoloji (%19,6)
  - Botanik (%16,7)
İlişkili Konular: Teknoloji (%22,7)

[Sohbet]
Genel sohbet konusu: Bilim + Biyoloji + Kitaplar
Tema: biyoloji hakkında bilimsel kitaplar
Arama sorgusu: "biyoloji hakkında bilimsel kitaplar"
```

Web araması açıkken her adımın altında `[İnternet Sonuçları]` bölümünde başlık, kısa özet ve
bağlantı listelenir; her adımda "Sonuçlar veritabanına kaydedildi." satırı da yazılır.

## Nasıl çalışır

```
Metin → Ön işleme (Türkçe küçük harf, normalizasyon, F5 kök)
      → TF-IDF (sözcük 1-2 gram + karakter 2-5 gram)
      → Softmax regresyon (39 "Genel > Alt" sınıfı) → Temperature scaling
      → Genel konu = alt konu olasılıklarının toplamı; güven eşiği → "Belirsiz"
      → Sohbet takibi → Tema ifadesi → Arama sorgusu → Wikipedia / DuckDuckGo → SQLite
```

Kendi yazılan bileşenler:

| Dosya | İçerik |
|---|---|
| `src/preprocessing/text.py` | Türkçe küçük harf (I/İ), normalizasyon, tokenizasyon, stopword, F5 kökleme |
| `src/ml/vectorizer.py` | TF-IDF: n-gram çıkarma, yumuşatılmış IDF, alt doğrusal TF, L2 norm |
| `src/ml/naive_bayes.py` | Laplace düzeltmeli çok terimli Naive Bayes (temel model) |
| `src/ml/softmax_regression.py` | Sınıf ağırlıklı softmax regresyon, mini-batch Adam, L2, erken durdurma |
| `src/ml/calibration.py` | Temperature scaling (altın oran aramasıyla NLL en küçükleme) |
| `src/ml/metrics.py` | Doğruluk, precision/recall/F1, karmaşıklık matrisi, ECE |
| `src/ml/splitting.py` | Terim gruplu, konu dağılımını koruyan veri bölme |

Eğitimde Naive Bayes ve softmax regresyon doğrulama setinde karşılaştırılır ve iyi olan
seçilir (softmax 0,854 / Naive Bayes 0,833 macro F1).

## Veri

* **Türkçe Tabu Veri Seti** ([wleeaf/taboo-dataset](https://github.com/wleeaf/taboo-dataset),
  MIT, commit `f621b46`): 150 kategoride 37.278 kavram kartı. 41 kategori taksonomiye eşlendi,
  birden fazla konuya girebilecek 31 kategori çıkarıldı, kalan kategoriler "Diğer" oldu.
* Konu ve alt konu adlarının kendisi ("biyoloji", "Kuantum Mekaniği" gibi) kısa eğitim
  örnekleri olarak eklendi; Tabu kartlarında kartın kendi kategori adı yasaklı sözcük olduğu
  için bu adlar veride neredeyse hiç geçmiyordu.
* Harici değerlendirme: Turkish BQuAD (biyoloji paragrafları) ve Turkish Reading Comprehension
  QA (Osmanlı tarihi paragrafları), ikisi de MIT lisanslı.
* Bölme terim gruplu yapılır; eğitim ile test arasında ortak terim ve birebir aynı metin
  yoktur. "Diğer" kategorilerinin %35'i eğitimden tamamen ayrılır: yarısı güven eşiğini
  seçmek, yarısı yalnızca son ölçüm için kullanılır.

Ham veri repoda tutulmaz; `python proje.py` ilk çalıştırmada indirir.

## Sonuçlar

Test seti bir kez, eğitim ve eşik seçimi bittikten sonra ölçüldü
(`reports/degerlendirme_raporu.json`):

| Ölçüm | Değer |
|---|---|
| Genel konu macro F1 (güven eşiğiyle, 3.479 örnek) | 0,830 |
| Genel konu macro F1 (eşiksiz en olası konu) | 0,850 |
| Doğruluk (eşikli / eşiksiz) | 0,882 / 0,898 |
| Alt konu doğruluğu (genel konu doğruyken) | 0,931 |
| Kalibrasyon hatası (ECE, 15 aralık) | 0,022 |
| Eğitimde hiç görülmemiş konuları reddetme oranı (3.275 örnek) | 0,884 |
| Harici biyoloji paragrafları (en olası konu = Biyoloji) | 0,688 |
| Harici Osmanlı tarihi paragrafları (en olası konu = Tarih) | 0,420 |
| Tahmin süresi (tek metin, medyan) | 0,7 ms |

Ödev senaryoları ve el ile yazılmış 5 kabul cümlesinden 6/7'si geçer. Kalan durum ve diğer
sınırlılıklar [docs/RAPOR.md](docs/RAPOR.md) içinde anlatılmıştır.

## Testler

```bash
pip install -r requirements-dev.txt
python -m pytest            # 164 geçer, 1 canlı ağ testi atlanır, 1 bilinen sorun (xfail)
python -m ruff check .
python -m mypy src proje.py
```

Model gerektiren testler için önce `python proje.py` ile modelin eğitilmiş olması gerekir.
Canlı Wikipedia testi `NLP_NETWORK_TESTS=1` ile çalışır.

## Proje yapısı

```
proje.py                   Tek giriş noktası: konsol uygulaması, eğitim ve değerlendirme
src/config.py              Ayarlar
src/preprocessing/text.py  Türkçe metin ön işleme
src/ml/                    Kendi yazılan ML algoritmaları
src/data/                  Veri indirme ve veri seti hazırlama
src/models/                Taksonomi, konu modeli, eğitim ve değerlendirme
src/services/              Sohbet takibi, sorgu üretimi, web arama, uygulama servisi
src/database/db.py         SQLite katmanı
tests/                     Testler
reports/                   Eğitim ve değerlendirme raporları (JSON)
docs/RAPOR.md              Proje raporu
```

## Veritabanı

Varsayılan dosya `nlp_app.db`. Tablolar: `metinler` (metin, genel konu, güven, alt konular,
tüm olasılıklar), `sohbet_konulari` (sohbetin genel konusu ve arama sorgusu),
`arama_sonuclari` (başlık, özet, bağlantı), `sessions`, `model_metadata`, `search_cache`.

```bash
sqlite3 nlp_app.db "SELECT message_index, general_topic, confidence FROM metinler ORDER BY id DESC LIMIT 5;"
```

## Lisanslar ve kaynaklar

| Kaynak | Lisans |
|---|---|
| Türkçe Tabu Veri Seti (wleeaf/taboo-dataset) | MIT, Copyright (c) 2026 |
| Turkish BQuAD (TurQuest/turkish-bquad) | MIT, Copyright (c) 2021 TurQuest |
| Turkish Reading Comprehension QA (okanvk) | MIT, Copyright (c) 2020 Okan |
| numpy, scipy | BSD-3-Clause |
| requests | Apache-2.0 |

Kaydedilen model Tabu verisinden türetilmiş n-gram sözlüğü içerir; model dağıtılırsa yukarıdaki
bildirim de eklenmelidir. Wikipedia içerikleri CC BY-SA 4.0 lisanslıdır ve kaynak bağlantısıyla
birlikte gösterilir.
