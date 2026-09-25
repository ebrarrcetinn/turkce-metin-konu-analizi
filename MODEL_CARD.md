# Model Card — turkish-topic-e11-flat-calibrated v1.0.0

| Alan | Değer |
|---|---|
| Model adı | `turkish-topic-e11-flat-calibrated` |
| Sürüm | 1.0.0 (`src/config.py: MODEL_VERSION`) |
| Artifact | `models/topic_model.joblib` (22,6 MB) + `models/topic_model.json` (metadata, SHA-256) |
| Eğitim tarihi | `models/topic_model.json → training_timestamp` (UTC) |
| Veri sürümü | `taboo@f621b46+split-v1` |
| Lisans (bileşenler) | THIRD_PARTY_NOTICES.md |

## Problem ve amaçlanan kullanım

Türkçe serbest metnin 8 genel konudan birine (Fizik, Kimya, Biyoloji, Teknoloji, Bilim,
Kitaplar, Spor, Tarih) ve 38 alt konudan birine atanması; taksonomi dışı metinlerin "Diğer" ya da
"Belirsiz" olarak işaretlenmesi. Amaçlanan kullanım: eğitim/demo amaçlı sohbet konusu takibi ve
ilgili ansiklopedik içerik araması.

**Amaçlanmayan kullanım:** içerik moderasyonu, kişi/grup hakkında karar, akademik
değerlendirme/notlama, tıbbi/hukuki sınıflandırma; taksonomi dışı alanlarda (sağlık, hukuk,
ekonomi…) konu tespiti.

## Mimari

```
metin → normalize (NFC, mojibake, HTML/URL/mention/emoji, Türkçe küçük harf)
      ├─ word 1-2gram TF-IDF (F5 kök, stopword) ─┐
      └─ char_wb 2-5gram TF-IDF ─────────────────┴→ LinearSVC (C=1, balanced, 39 sınıf)
      → softmax(skor / T)   [T=0,1592; ≤3 içerik sözcüklü girdilerde T_short=0,2081]
      → genel konu olasılığı = alt konu olasılıklarının toplamı
      → güven < 0,75 ise "Belirsiz"; en olası "Diğer" ise "taksonomi dışı"
```

Sınıflar: 38 "Genel > Alt" + "Diğer". Olasılıklar kalibre edilmiştir (temperature scaling).
Seçim gerekçesi: MODEL_REPORT.md; tüm deneyler: EXPERIMENTS.md.

## Eğitim

* Veri: train (16.319) ile model/eşik seçimi; son model train+val (19.583) ile yeniden eğitildi.
* Hiperparametreler: C=1,0 (ızgara 0,1–8), class_weight=balanced, T train içi 3-kat terim-gruplu
  OOF NLL ile; eşik 0,75 validation macro F1 ile (ızgara 0,20–0,90); alt konu eşiği 0,50.
* Tohum: `RANDOM_SEED=42`. Temiz ortamda iki kez yeniden üretildi; tüm metrikler birebir aynı
  (joblib baytları Python hash sıralaması nedeniyle farklı olabilir).
* Eğitim süresi: final fit 58,6 s, `train.py` toplam ~3 dk (2 vCPU).

## Metrikler (test, bir kez)

| Metrik | Değer |
|---|---:|
| Accuracy (eşikli) | 0,8943 |
| Macro precision / recall | 0,8752 / 0,8295 |
| **Macro F1 (eşikli)** | **0,8500** |
| Weighted F1 | 0,8934 |
| Macro F1 (argmax) | 0,8386 |
| ECE (15 bin) | 0,0374 |
| Alt konu yol doğruluğu | 0,847 |
| OOD reddetme (27 görülmemiş kategori) | 0,882 |
| Harici recall (BQuAD / Osmanlı, argmax) | 0,730 / 0,660 |

Sınıf bazında (test, eşikli): Spor F1 0,935; Diğer 0,917; Biyoloji 0,885; Teknoloji 0,852;
Tarih 0,838; Kitaplar 0,826; Fizik 0,820; Kimya 0,797; **Bilim 0,779** (n=36).

Kalibrasyon (test reliability): güven 0,9–1,0 aralığında 2.338 örnek, ortalama güven 0,985,
doğruluk 0,955; 0,8–0,9 → 0,855/0,857; 0,7–0,8 → 0,752/0,731. Düşük güven aralıklarında
(0,3–0,6) model hafifçe *düşük güvenli*dir. Grafik: `reports/figures/reliability_diagram.png`.

## Eşikler

`min_confidence=0,75`, `subtopic_threshold=0,50`, `short_input_temperature=0,2081`,
`per_class_min_confidence=None` (denendi, reddedildi).

## Donanım ve gecikme

CPU-only (GPU/CUDA gerekmez). 2 vCPU Xeon 2,8 GHz, 7,8 GB RAM, Python 3.11.15, scikit-learn 1.8.0:
yükleme 0,93 s; soğuk başlatma (import+yükleme+ilk tahmin) 2,3 s; tek metin p50 2,2 ms, p95 2,5 ms;
toplu 0,39 ms/metin; model belleği ~114 MB RSS.

## Sınırlamalar ve bilinen hata durumları

* Eğitim metinleri kısa sözlük tanımları → anlatı/sohbet üslubunda düşüş (harici recall
  0,66–0,73).
* "Bilim" sınıfı yalnızca epistemoloji: bilimsel yöntem cümleleri "Belirsiz"e düşebilir (KI-001).
* Kuantum Bilgisayarlar alt konusu 32 karta dayanır.
* Çok konulu metinlerde tek baskın konu; ikinci konu yalnızca olasılıklarda görünür.
* İroni, argo, yazım hatası, ASCII Türkçe ("Istanbul", "cok") eğitim verisinde yok.
* Dışlanan 31 komşu alan (anatomi, havacılık, felsefe, mitoloji…) için davranış tanımsız.
* Güvenle yanlış tahminlerin elle incelenen örneklemde %70'i meşru alan örtüşmesi, %20'si
  etiket gürültüsü (ERROR_ANALYSIS.md §3).

## Etik ve bias

* Veri tek bir açık kaynak projeden, büyük olasılıkla kısmen makine üretimi; Türkiye merkezli
  kültürel içerik (Osmanlı, Türk halk edebiyatı) baskın.
* Kişisel veri içermez; model kişiler hakkında çıkarım yapmaz.
* Çıkarım yereldir (kullanıcı metni üçüncü tarafa gönderilmez); yalnızca türetilen **arama
  sorgusu** Wikipedia/DuckDuckGo'ya gider (README "Gizlilik").
* Güven skorları kalibre edilmiş olsa da harici metinlerde kalibrasyon ölçülmedi.
