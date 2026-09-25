# Final Rapor — Türkçe NLP Konu Analiz Sistemi

Tarih: 2026-09-25 · Model: `turkish-topic-e11-flat-calibrated` v1.0.0 · Veri: `taboo@f621b46+split-v1`

Tüm sayılar `reports/*.json` dosyalarından alınmıştır. `python -m scripts.check_docs` anahtar
metriklerin bu raporla uyumunu otomatik denetler.

## 1. Executive Summary

Türkçe metni 8 genel konu / 38 alt konuya ayıran, taksonomi dışı ve belirsiz girdileri reddeden,
sohbet boyunca birleşik tema üreten, temadan arama sorgusu oluşturup Wikipedia'dan bilgi getiren
ve SQLite'a kaydeden bir sistem geliştirildi. 12 aday model validation'da karşılaştırıldı;
**F5 kök + karakter n-gram TF-IDF + LinearSVC** (temperature scaling ile kalibre) seçildi.

Test seti (bir kez ölçüldü, n=3.263): **macro F1 0,850**, doğruluk 0,894, weighted F1 0,893,
ECE 0,037, alt konu yol doğruluğu 0,847. Eğitimde hiç görülmemiş 27 kategoriden 6.060 metnin
%88,2'si reddedildi. Kritik kuantum testi 2/2, kabul testleri 2/3 (B: "Belirsiz", KI-001).
Test paketi: 140 geçti, 1 atlandı (canlı ağ), 1 beklenen başarısızlık. CPU'da tek metin p50 2,2 ms.

Başlıca sınırlamalar: eğitim metinleri kısa sözlük tanımları (insan yazımı anlatı metinlerinde
recall 0,66–0,73); "Bilim" sınıfının kaynağı yalnızca epistemoloji; ortam ağ politikası nedeniyle
Transformer denemesi ve canlı web testi yapılamadı.

## 2. Problem Definition

Girdi: Türkçe serbest metin (sohbet mesajı). Çıktı: genel konu + güven, alt konu(lar), sohbet
teması, arama sorgusu, web sonuçları. Görev tipi: çok sınıflı (tek etiket) hiyerarşik
sınıflandırma + reddetme (OOD/belirsizlik) + sohbet düzeyinde birikimli konu modeli.

## 3. Requirements

Şartnamenin 87 maddesi 28 faza ayrılıp dört kontrol noktasında (madde 20, 40, 60 ve final)
temiz ortamda uçtan uca test edildi (PROJECT_STATE.md). İşlevsel gereksinimler: §2 (11 madde),
§21 kuantum, §22 OOD, §23–25 sohbet/oturum, §26–28 web ve DB, §38–40 kabul ve uç durumlar.
Kalite gereksinimleri: sızıntısız bölme, ölçülmüş model seçimi, kalibrasyon, testler,
tekrar üretilebilirlik, dokümantasyon.

## 4. Dataset Research

11 aday incelendi (DATASET_RESEARCH.md). Kaggle/UCI/Hugging Face ortamdan erişilemediği için
dokümantasyon üzerinden değerlendirildi; GitHub adayları klonlanıp doğrudan incelendi. Haber
veri setleri Fizik/Kimya/Biyoloji ayrımı ve alt konu sağlamadığı, TurkishMMLU kontrollü erişimli
olduğu, TQuAD lisanssız olduğu için elendi.

## 5. Selected Dataset

Türkçe Tabu Veri Seti (MIT, 37.278 kart, 150 kategori, commit `f621b46`). 41 kategori
taksonomiye eşlendi, 31 komşu kategori dışlandı, 78 kategori "Diğer" oldu (27'si yalnızca OOD
testinde). Harici değerlendirme: Turkish BQuAD (biyoloji) ve Osmanlı Tarihi RC (MIT), 400'er
cümle. Provenance: sabit commit + dosya bazında SHA-256 (DATASET_CARD.md).

## 6. Exploratory Data Analysis

* Metinler 5–13 sözcük (ort. 10,7); harici setler 6–38 (ort. 13,8).
* Sözlük 48.307 yüzey biçimi; bir kez geçen sözcük oranı %52,3 → morfoloji için karakter n-gram
  ve F5 kök gerekçesi.
* Taksonomi içi dengesizlik 9,0 (Spor 2.247 / Bilim 249); "Diğer" %55,5.
* URL/emoji/mention/hashtag/mojibake 0; 1 HTML benzeri (Dirac notasyonu, yanlış pozitif).
* Elle inceleme (60 kart): %28 terim–tanım uyuşmazlığı, %27 kesik tanım, kategori düzeyinde açık
  hata 0. Grafikler: `reports/figures/` (etiket, alt konu, uzunluk, sık sözcükler).

## 7. Data Cleaning

Boş tanım 0; birebir tekrar 0; çelişkili etiket 0; dışlanan komşu kategori 7.693 kart.
Temiz veri 29.585. Bölme: 70/15/15, terim-gruplu stratified, seed 42. Sızıntı: grup, birebir
metin ve kosinüs ≥ 0,9 yakın-kopya çakışması train–val/test/harici arasında **0**.

## 8. NLP Preprocessing

`normalize`: mojibake onarımı (cp1252/latin-1) → NFC → HTML/URL/mention temizliği → hashtag
sözcüğü korunur → apostrof ekleri atılır ("Ankara'da"→"ankara") → emoji temizliği → **Türkçe
küçük harf** (I→ı, İ→i; Python `lower()` hatalı) → rakam/noktalama temizliği. Sözcük yolu:
stopword + F5 kök (ilk 5 harf); karakter yolu: `char_wb` 2–5gram (kök gerektirmez). F5, sözcük
modelinde +3,3 macro F1 puanı kazandırdı (E04→E05). Lemmatizasyon/morfolojik analizör
kullanılmadı: ek bağımlılık gerektirir, karakter n-gram + F5 ile sözlük dışı oranı zaten ~%70
düşüyor (ERROR_ANALYSIS §4). Aynı fonksiyon eğitim ve çıkarımda kullanılır.

## 9. Taxonomy

8 genel, 38 alt konu (TAXONOMY.md). Şartname taksonomisi veriye zorla uydurulmadı; örneğin
"Bilimsel Kitaplar" bir sınıf değil, sohbet katmanında Kitaplar + Bilim birleşimidir. Kuantum
bilgisayar kartları yalnızca terim alanına bakan şeffaf bir kuralla Teknoloji'ye ayrıldı
(32 kart).

## 10. Models Considered

Naive Bayes (Multinomial, Complement), Logistic Regression, Linear SVM, SGD (modified Huber),
sözcük / F5 / karakter / birleşik TF-IDF, LSA-300 yoğun gömme + MLP, fastText supervised;
hiyerarşik ve düz alt konu modelleri; sigmoid / isotonic / temperature kalibrasyonu. BERTurk,
XLM-R ve Sentence-Transformers değerlendirildi ama **ölçülemedi** (huggingface.co engelli;
ADR-006).

## 11. Experiments

EXPERIMENTS.md'de 20 deney ve 5 turluk hata-analizi döngüsü var. Özet (val macro F1): E01 NB
0,744 · E04 SVM 0,784 · E05 SVM+F5 0,817 · E06 char SVM 0,811 · E07 word+char SVM 0,827 ·
E08 word+char LR 0,828 · E10 LSA+MLP 0,695 · E12 fastText 0,707 · **E11 0,836**.
5-kat grup CV: E11 0,831 ± 0,010; E07 0,827 ± 0,009; E08 0,821 ± 0,007.

## 12. Hyperparameter Search

Manuel kontrollü ızgara (küçük veri; Optuna gereksiz): C ∈ {0,1 … 8} × "Diğer" alt-örneklemesi
∈ {yok, 4000} → C=1,0 (0,5–2 plato), alt-örnekleme her C'de zararlı. Güven eşiği 0,20–0,70
ızgarasında sınırda çıktı → 0,90'a genişletildi → iç optimum 0,75. Alt konu eşiği 0,50. Sıcaklık
T grup-OOF NLL ile (0,159); kısa girdi T_short 0,208. Tüm seçimler train/val üzerinde; test
kullanılmadı.

## 13. Final Model

F5 word 1-2gram + char_wb 2-5gram TF-IDF → LinearSVC (C=1, balanced) → 39 birleşik sınıf →
temperature scaling → genel = Σ alt → eşik 0,75. Train+val (19.583) ile yeniden eğitildi
(final fit 58,6 s). Artifact 22,6 MB + SHA-256'lı metadata (MODEL_CARD.md).

## 14. Evaluation

| Set | Metrik | Değer |
|---|---|---:|
| Test (eşikli) | Accuracy / Macro P / Macro R | 0,894 / 0,875 / 0,830 |
| Test (eşikli) | **Macro F1** / Weighted F1 | **0,850** / 0,893 |
| Test (argmax) | Macro F1 / Accuracy | 0,839 / 0,883 |
| Test | ECE (15 bin) | 0,037 |
| Test | Alt konu yol doğruluğu | 0,847 |
| Test | "Belirsiz" / "Diğer" oranı | %8,4 / %50,4 |
| OOD (27 görülmemiş kategori) | Reddetme / yanlış kabul | 0,882 / 0,118 |
| Harici BQuAD (Biyoloji) | Recall argmax / eşikli | 0,730 / 0,573 |
| Harici Osmanlı (Tarih) | Recall argmax / eşikli | 0,660 / 0,445 |
| Kısa girdi (test terimleri) | Doğruluk / ECE (T_short) | 0,668 / 0,070 |

Sınıf bazında (eşikli F1): Spor 0,935 · Diğer 0,917 · Biyoloji 0,885 · Teknoloji 0,852 ·
Tarih 0,838 · Kitaplar 0,826 · Fizik 0,820 · Kimya 0,797 · Bilim 0,779.
Val (0,849) ≈ test (0,850): validation'a aşırı uyum yok. Train≈1,0 ile val 0,84 arasındaki fark
yüksek kapasiteli seyrek modelin kısa metinleri ezberlemesi; C taraması düzenlileştirmenin val'i
iyileştirmediğini gösterdi.

Kalibrasyon: güven 0,9–1,0 aralığında doğruluk 0,955 (ort. güven 0,985); 0,8–0,9 → 0,857 (0,855);
0,7–0,8 → 0,731 (0,752). "%90 güven ≈ %90 doğruluk" yüksek güvende büyük ölçüde sağlanıyor; orta
güven aralığında model hafif düşük güvenli. `reports/figures/reliability_diagram.png`.

## 15. Confusion Matrix

`reports/figures/confusion_matrix_test.png` (eşikli). Baskın hata taksonomi içi ↔ "Diğer"
sınırında: satır "Diğer" sütunu Fizik 25, Kimya 12, Biyoloji 33, Teknoloji 37, Bilim 6, Kitaplar
31, Spor 30, Tarih 35; "Diğer" satırında Teknoloji 22, Biyoloji 20, Fizik 15. Taksonomi içi
karışma küçük (en büyük: Kimya→Fizik 5, Teknoloji→Fizik 4).

## 16. Error Analysis

ERROR_ANALYSIS.md. 345 hatanın 134'ü "Belirsiz", 75'i "Diğer" (reddetme), 136'sı güvenle yanlış;
147'sinde doğru sınıf 2. sırada. Güvenle yanlış 30 örneğin elle incelemesinde %70 meşru alan
örtüşmesi (albedo → Fizik, ultrason → Fizik), %20 etiket gürültüsü, %10 gerçek model hatası.
Harici metinlerde anlatı cümleleri ("Ama Cem Sultan bu teklifi reddetti…") "Diğer"e kayıyor;
alan/üslup kayması en önemli risk. İroni modellenmiyor. Kısa girdiler en zayıf durum
(doğruluk 0,67).

## 17. Conversation Tracking

`skor_t = skor_{t-1}·0,7 + p_t` (genel ve alt konu için ayrı; "Diğer" ayrı birikir ve
baskınsa tema "Belirsiz" olur). Tema: payı ≥ %20 olan en fazla 3 konu; tek konuda alt konu payı
≥ %50 ise alt konuya daraltma. İfade üretimi rol tabanlı (ortam / niteleyici / alan).

DECAY deneyi (400 simüle sohbet): 0,7, üç konulu birikimli daraltmayı koruyan en küçük değer
(korunum 0,6'da %6 → 0,7'de %56); 0,5 anlık izlemede daha iyi (0,922 ile 0,834) ama §39'u
karşılamıyor. Gerçek model çıktılarıyla §39: "Dün akşam sürükleyici bir roman okudum." →
"kitaplar"; "Bilimsel yöntem ve deneyler hakkında düşünüyorum." → "bilimsel kitaplar";
"Hücreler, DNA ve evrim konusu beni çok etkiliyor." → "biyoloji hakkında bilimsel kitaplar"
(test: `test_conversation_acceptance_books_science_biology`). Şartnamenin A/B/C cümleleriyle
ikinci adım "bilimsel kitaplar"a ulaşıyor, üçüncü adımda B'nin düşük güveni nedeniyle
"biyoloji kitapları" oluyor.

## 18. Web Search

Sorgu = tema çekirdeği + son mesajdan ≤2 anahtar sözcük (SVM ağırlığı ≥ 0,30; çekimli fiiller
elenir, hal ekleri kırpılır), örn. "kuantum bilgisayarlar kübit". Wikipedia `generator=search` +
`extracts|info` (başlık, 2 cümlelik özet, URL) → sonuç yoksa/hata varsa DuckDuckGo Instant
Answer. Timeout 6 s, 2 retry, üstel geri çekilme, `Retry-After` (≤5 s), 4xx/5xx/bozuk JSON/boş
yanıt ayrı durumlar, UTF-8 zorlama, HTML temizliği, yalnızca `https` + izinli alan adları.
Bağlantı yoksa 120 s devre kesici (ilk çevrimdışı istek 6,1 s; sonraki turlar 8 ms). SQLite
önbellek 24 s. **Canlı API bu ortamda test edilemedi** (KI-002); ayrıştırıcılar API
dokümantasyonuna göre hazırlanmış örnek yanıtlarla, istemci sahte HTTP oturumuyla, çevrimdışı
davranış gerçek ağ hatasıyla doğrulandı.

## 19. Database Design

SQLite v1 şeması (ARCHITECTURE.md ER diyagramı): `sessions`, `model_metadata`, `metinler`,
`sohbet_konulari`, `arama_sonuclari`, `search_cache`. Foreign key açık, 5 indeks, her yazma tek
transaction, tüm sorgular parametreli (iki f-string: sabit PRAGMA ve beyaz listeli tablo adı),
`busy_timeout` 5 s, `PRAGMA user_version` migration (daha yeni şema reddedilir). Her tahmin
üreten model sürümüne bağlı. `sıfırla` yeni oturum açar, kayıtları silmez.

## 20. Software Architecture

Eğitim ve çıkarım ayrı: `download_data → build_dataset → train → evaluate` hattı artifact
üretir; `proje.py` yalnızca yükler. Modüller: preprocessing, data, models (taxonomy, pipeline,
calibration, training, artifact, predictor), evaluation, services (conversation, query_builder,
web_search, app), database. Tüm sabitler `src/config.py`'de; `logging` (konsol ERROR, dosya
INFO). ChatService dış bağımlılık hatalarını izole eder (ARCHITECTURE.md).

## 21. Testing

142 test (TEST_REPORT.md, betikle gerçek çalıştırmadan üretildi): 140 PASS, 1 SKIP (canlı ağ),
1 XFAIL (KI-001, strict). Kapsam: Türkçe küçük harf, normalizasyon, mojibake, tokenizasyon,
boş/anlamsız girdi, taksonomi ve hiyerarşi, karar/eşik mantığı, kalibrasyon, artifact
kurcalama/eksik/bozuk metadata, decay formülü, sıfırlama, tema ifadeleri, sorgu üretimi,
Wikipedia/DDG ayrıştırıcıları, URL doğrulama, retry/Retry-After/timeout/404/bozuk JSON/boş
sonuç, DB şema/migration/FK/SQL injection/önbellek TTL/yazma hatası, uçtan uca zincir, önbellek,
devre kesici, DB hatasında devam, CLI komutları/EOF/eksik model, kuantum, kabul, §40 uç
durumları. Mutasyon kontrolü: decay formülü ve Türkçe küçük harf kuralı bozulduğunda ilgili
testler başarısız oldu. `ruff` (0.8.6 ve 0.15) ve `mypy` temiz. Dört kontrol noktasında temiz
kopyadan tüm hat yeniden çalıştırıldı; metrikler birebir aynı.

## 22. Performance

2 vCPU, GPU yok (`reports/performance.json`): soğuk başlatma (import + yükleme + ilk tahmin)
2,26 s; model yükleme 0,93 s; model belleği ~114 MB RSS; ön işleme 0,05 ms/metin; tek metin
çıkarım p50 2,23 ms / p95 2,52 ms; toplu 0,39 ms/metin; web'siz tam tur (DB dahil) p50 5,2 ms;
önbellek okuma 0,016 ms. Canlı web isteği süresi ölçülemedi (ağ kısıtı).

## 23. Security

| Konu | Durum |
|---|---|
| SQL injection | Parametreli sorgular; injection metni literal saklanıyor (test) |
| Güvensiz serileştirme | joblib yalnızca SHA-256 + şema + sınıf doğrulamasından sonra yüklenir |
| Arbitrary file write / path traversal | İndirme hedefleri `data/raw` dışına çıkamaz (kontrol); tar açma kaldırıldı; artifact yazımı atomik |
| Güvenilmeyen HTML | Web özetleri etiketlerden arındırılıp unescape edilir, uzunluk sınırlı |
| HTTP timeout / URL doğrulama | Her istekte timeout; yalnızca https + wikipedia.org/duckduckgo.com |
| Bağımlılık açıkları | `pip-audit`: requirements.txt temiz; pytest 8.3.5'teki PYSEC-2026-1845 için 9.0.3'e yükseltildi |
| Sırlar | API anahtarı yok; kaynakta sır kalıbı taraması temiz |
| Kod çalıştırma | `eval/exec/shell=True/os.system` yok |

## 24. Limitations

KNOWN_ISSUES.md: KI-001 Bilim sınıfı kapsamı; KI-002 canlı web testi; KI-003 alan/üslup
kayması; KI-004 Transformer yok; KI-005 kaynak veri gürültüsü; KI-006 kısa/ASCII Türkçe girdi.
Ayrıca tek etiketli eğitim verisi gerçek multi-label öğrenmeyi engelliyor; ironi/argo yok.

## 25. Future Improvements

1. BERTurk ince ayarı ve cümle gömme + LR adaylarını aynı bölmelerle ölçmek.
2. Bilim/Kitaplar için insan yazımı, açık lisanslı ek veri (ör. Vikipedi kategori makaleleri).
3. Sohbet üslubunda etiketli test seti; harici kalibrasyon ölçümü.
4. Deasciifier ve yazım hatası dayanıklılığı.
5. Multi-label etiketleme ve eşik öğrenimi.
6. Canlı Wikipedia entegrasyon testi CI'da.

## 26. Conclusion

Proje, gerçek ve açık lisanslı veriyle, sızıntısız bir protokolle seçilmiş ve kalibre edilmiş bir
model üzerine kurulu, test edilmiş ve tekrar üretilebilir bir sistem olarak tamamlandı. Test
performansı (macro F1 0,850) ve OOD reddetme (%88,2) güçlü. Ancak harici metinlerdeki düşüş
gösteriyor ki gerçek sohbet mesajlarında başarı daha düşük olacaktır. Açık kalan iki madde
(kabul testi B ve canlı web testi) nedenleriyle raporlandı; ikisi de veri/ortam kısıtından
kaynaklanıyor ve çözüm yolları KNOWN_ISSUES.md'de tanımlı.

### Donanım ve yazılım (§65)

| | |
|---|---|
| OS | Linux 6.18 x86_64 (glibc 2.39) |
| Python | 3.11.15 |
| CPU | Intel Xeon @ 2.80 GHz, 2 vCPU |
| RAM | 7,8 GB |
| GPU / CUDA | Yok / yok (gerekmez; model CPU'da çalışır) |
| Framework | scikit-learn 1.8.0, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, joblib 1.5.3, requests 2.33.1 |

### Kaynaklar

* Can, F. vd. (2008). *Information retrieval on Turkish texts.* JASIST 59(3) — F5 önek kökleme.
* Guo, C. vd. (2017). *On Calibration of Modern Neural Networks.* ICML — temperature scaling.
* Yüksel, A. vd. (2024). *TurkishMMLU.* Findings of EMNLP — aday veri seti değerlendirmesi.
* scikit-learn belgeleri: `LinearSVC`, `CalibratedClassifierCV`, `StratifiedGroupKFold`.
* MediaWiki Action API (`generator=search`, `prop=extracts`), DuckDuckGo Instant Answer API.
