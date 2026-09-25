# Türkçe NLP Konu Analiz Sistemi

Türkçe serbest metni **genel konu** ve **alt konu** düzeyinde sınıflandıran, sohbet boyunca
konuyu takip edip birleşik bir tema üreten, bu temadan arama sorgusu oluşturup Türkçe
Wikipedia'dan bilgi getiren ve her şeyi SQLite'a kaydeden konsol uygulaması.

> Test seti (bir kez ölçüldü): **macro F1 0,850**, doğruluk 0,894, ECE 0,037; eğitimde hiç
> görülmemiş taksonomi dışı metinlerin %88,2'si doğru biçimde reddediliyor. Ayrıntı:
> [MODEL_CARD.md](MODEL_CARD.md), [FINAL_REPORT.md](FINAL_REPORT.md).

## 1. Proje amacı

Metodolojik olarak savunulabilir, ölçülmüş, test edilmiş ve tekrar üretilebilir bir Türkçe
konu sınıflandırma + sohbet teması + bilgi getirme sistemi.

## 2. Özellikler

* 8 genel konu (Fizik, Kimya, Biyoloji, Teknoloji, Bilim, Kitaplar, Spor, Tarih), 38 alt konu;
  taksonomi dışı ("Diğer") ve düşük güvenli ("Belirsiz") girdiler için reddetme.
* Kalibre edilmiş güven skorları (temperature scaling; kısa girdiler için ayrı sıcaklık).
* "Kuantum dolanıklık…" → Fizik > Kuantum Mekaniği; "Kuantum işlemciler kübit…" → Teknoloji >
  Kuantum Bilgisayarlar.
* Sohbet takibi (`skor = eski·0,7 + olasılık`) ve rol tabanlı tema birleştirme:
  Kitaplar → "bilimsel kitaplar" → "biyoloji hakkında bilimsel kitaplar".
* Model ağırlıklarından anahtar sözcük seçen arama sorgusu; Wikipedia (birincil) → DuckDuckGo
  (yedek), API anahtarı yok; timeout, retry, devre kesici, 24 saatlik SQLite önbellek.
* İnternet ya da veritabanı olmasa bile sınıflandırma ve sohbet takibi çalışır.
* SQLite: oturumlar, metinler, sohbet konuları, arama sonuçları, model sürümü; migration.

## 3. Mimari

```
Girdi → Ön işleme → Sınıflandırıcı (F5 word + char TF-IDF → LinearSVC → T-scaling)
      → Hiyerarşi (genel = Σ alt) → Sohbet takibi → Tema → Sorgu → Önbellek/Web → SQLite
```

Mermaid diyagramları ve modül tablosu: [ARCHITECTURE.md](ARCHITECTURE.md).

## 4. Kullanılan veri seti

[Türkçe Tabu Veri Seti](https://github.com/wleeaf/taboo-dataset) (MIT, 37.278 kart, 150
kategori), 41 kategori taksonomiye eşlendi, 31 komşu kategori dışlandı, kalanı "Diğer".
Harici değerlendirme: Turkish BQuAD (biyoloji) ve Osmanlı Tarihi RC (MIT). Araştırma ve
gerekçe: [DATASET_RESEARCH.md](DATASET_RESEARCH.md); kart: [DATASET_CARD.md](DATASET_CARD.md);
eşleme: [TAXONOMY.md](TAXONOMY.md).

## 5. Kullanılan model

`turkish-topic-e11-flat-calibrated` v1.0.0: F5 kökleme ile sözcük 1-2gram + karakter 2-5gram
TF-IDF → LinearSVC (C=1, dengeli sınıf ağırlığı), 39 birleşik "Genel > Alt" sınıfı,
temperature scaling (T=0,159), güven eşiği 0,75. CPU'da tek metin ~2,2 ms, artifact 22,6 MB.

## 6. Model seçim gerekçesi

12 aday (Naive Bayes, LR, SVM, SGD, karakter/sözcük/F5 TF-IDF, LSA gömme + MLP, fastText)
validation'da karşılaştırıldı; E11 en yüksek val macro F1 (0,836) ve 5-kat CV (0,831 ± 0,010)
verdi. Düz birleşik etiketli model, hiyerarşik iki aşamalı modelden alt konuda 5,7 puan daha iyi.
Transformer'lar ortamda Hugging Face erişimi olmadığı için ölçülemedi. Tablo:
[MODEL_REPORT.md](MODEL_REPORT.md); tüm deneyler: [EXPERIMENTS.md](EXPERIMENTS.md); kararlar:
[DECISIONS.md](DECISIONS.md).

## 7. Kurulum

Python 3.11+ (3.11.15 ile test edildi).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # çalışma zamanı + eğitim
pip install -r requirements-dev.txt                    # testler, lint, deneyler (opsiyonel)
```

## 8. Çalıştırma

Model artifact'ı repoda yoktur (22 MB, `.gitignore`); bir kez üretin (bkz. §9), sonra:

```bash
python proje.py              # web araması açık
python proje.py --no-web     # çevrimdışı mod
python proje.py --debug      # ayrıntılı log (stderr); her durumda logs/app.log
python proje.py --db baska.db --model models/topic_model.joblib
```

Ortam değişkenleri: `NLP_WEB_SEARCH=0`, `NLP_DB_PATH`, `NLP_MODEL_PATH`, `NLP_DECAY`,
`NLP_WEB_TIMEOUT`, `NLP_LOG_LEVEL`.

## 9. Training

```bash
python -m scripts.download_data     # sabit commit'lerden indirir, SHA-256 doğrular (~16 MB)
python -m scripts.build_dataset     # eşleme, temizlik, terim-gruplu bölme, sızıntı raporu
python train.py                     # kalibrasyon + eşik seçimi + final model (~3 dk, 2 vCPU)
```

İsteğe bağlı analiz ve deneyler:

```bash
python -m scripts.eda                               # reports/eda.json + reports/figures/
python -m scripts.run_experiments compare           # E01-E11 (validation)
python -m scripts.run_experiments fasttext          # E12 (requirements-dev gerekir)
python -m scripts.run_experiments cv                # 5-kat grup CV
python -m scripts.run_experiments tune --exp E11    # C ızgarası
python -m scripts.run_experiments hierarchy --exp E11 --C 1
python -m scripts.tune_decay                        # sohbet DECAY deneyi
```

## 10. Evaluation

```bash
python evaluate.py                  # test, OOD, harici, kuantum/kabul, performans
python -m scripts.error_analysis    # reports/error_analysis.json
python -m scripts.benchmark         # reports/performance.json (gecikme, bellek, donanım)
```

## 11. Test

```bash
python -m pytest                          # 142 test (model testleri artifact gerektirir)
NLP_NETWORK_TESTS=1 python -m pytest -m network   # canlı Wikipedia testi
python -m ruff check . && python -m mypy src scripts proje.py train.py evaluate.py
python -m scripts.make_test_report        # testleri çalıştırıp TEST_REPORT.md üretir
```

Son sonuç: 140 geçti, 1 atlandı (canlı ağ), 1 beklenen başarısızlık (KI-001) —
[TEST_REPORT.md](TEST_REPORT.md).

## 12. Konsol komutları

| Komut | Etki |
|---|---|
| `geçmiş` | Bu oturumdaki mesajlar, sınıflandırmalar ve sohbet konusu |
| `sıfırla` | Yeni sohbet bağlamı ve yeni `session_id`; eski kayıtlar DB'de kalır |
| `yardım` | Komut listesi |
| `çıkış` / `q` | Çıkış (Ctrl+D ve Ctrl+C da güvenle kapatır) |

## 13. Veritabanı

Varsayılan `nlp_app.db` (SQLite). Tablolar: `metinler`, `sohbet_konulari`, `arama_sonuclari`,
`sessions`, `model_metadata`, `search_cache`. Foreign key, indeks, transaction, parametreli
sorgular, `PRAGMA user_version` ile migration. Şema: [ARCHITECTURE.md](ARCHITECTURE.md).

```bash
sqlite3 nlp_app.db "SELECT message_index, general_topic, confidence FROM metinler ORDER BY id DESC LIMIT 5;"
```

## 14. Örnek çıktı (gerçek çalıştırma, `--no-web`)

Girdi: `Kuantum bilgisayarlar kübit kullanır.` ardından `q` (stdin'den verildiği için girdi
satırı ekranda görünmüyor):

```
Türkçe NLP Konu Analiz Sistemi
Model yükleniyor...
Model hazır (v1.0.0).
Web araması kapalı (çevrimdışı mod).
Komutlar: 'geçmiş' (bu oturumun mesajları), 'sıfırla' (yeni sohbet bağlamı; kayıtlar silinmez), 'yardım', 'çıkış' veya 'q'.

Metin girin:

[Metin Analizi]
Genel Konu: Teknoloji
Güven: %100,0
Alt Konular:
  - Kuantum Bilgisayarlar (%100,0)

[Sohbet]
Genel sohbet konusu: Teknoloji > Kuantum Bilgisayarlar
Tema: kuantum bilgisayarlar
Arama sorgusu: "kuantum bilgisayarlar kübit"

Sonuçlar veritabanına kaydedildi.

Metin girin:
Güle güle!
```

Web açıkken ve internet varsa `[İnternet Sonuçları]` altında başlık, özet ve URL listelenir;
internet yoksa "İnternete erişilemedi; sonuçlar gösterilemiyor (analiz kaydedildi)." yazılır.

## 15. Proje dizini

```
├── proje.py / train.py / evaluate.py      giriş noktaları
├── src/
│   ├── config.py, logging_setup.py
│   ├── preprocessing/text.py
│   ├── data/ (sources.py, build.py, taboo_manifest.json)
│   ├── models/ (taxonomy, pipeline, calibration, training, artifact, predictor)
│   ├── evaluation/metrics.py
│   ├── services/ (conversation, query_builder, web_search, app)
│   └── database/db.py
├── scripts/  download_data, build_dataset, eda, run_experiments, tune_decay,
│             error_analysis, benchmark, make_test_report, check_docs
├── tests/    (+ fixtures/)
├── reports/  eda.json, experiments.json, training_report.json, evaluation.json,
│             error_analysis.json, decay_experiment.json, performance.json, figures/
├── data/     raw/ processed/ (git dışı)   models/ (artifact git dışı)
└── *.md      README, ARCHITECTURE, DATASET_*, TAXONOMY, MODEL_*, EXPERIMENTS,
              ERROR_ANALYSIS, TEST_REPORT, FINAL_REPORT, DECISIONS, KNOWN_ISSUES,
              PROJECT_STATE, THIRD_PARTY_NOTICES
```

## 16. Bilinen sınırlamalar

* Eğitim metinleri kısa sözlük tanımları → insan yazımı anlatı metinlerinde recall 0,66–0,73.
* "Bilim" sınıfı yalnızca bilgi felsefesi kartlarından; bilimsel yöntem cümleleri "Belirsiz"e
  düşebilir (kabul testi B).
* Canlı web araması geliştirme ortamında test edilemedi (ağ politikası).
* Transformer karşılaştırması yapılamadı (Hugging Face erişimi yok).
* Tümü: [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**Gizlilik:** Sınıflandırma tamamen yereldir; kullanıcı mesajları üçüncü taraflara gönderilmez.
Web araması açıkken yalnızca üretilen **arama sorgusu** (tema + en fazla iki anahtar sözcük)
Wikipedia/DuckDuckGo'ya gider. Mesajlar yerel SQLite dosyasında ve `logs/app.log`'da (sorgular)
saklanır; `--no-web` ile hiçbir veri dışarı çıkmaz.

## 17. Gelecek geliştirmeler

1. BERTurk / çok dilli cümle gömmeleri ile aynı bölmelerde karşılaştırma (KI-004).
2. "Bilim" için bilimsel yöntem ve bilim tarihi içeren açık lisanslı veri (KI-001).
3. Sohbet üslubunda etiketli değerlendirme seti; harici kalibrasyon ölçümü.
4. ASCII Türkçe için deasciifier; ironi/argo için veri.
5. Çok konulu metinler için gerçek multi-label etiketleme.
