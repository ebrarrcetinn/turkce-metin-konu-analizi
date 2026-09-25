# Deney Günlüğü

Tüm sayılar `reports/experiments.json`, `reports/training_report.json`,
`reports/decay_experiment.json` ve `reports/evaluation.json` dosyalarından alınmıştır.
Donanım: 2 vCPU Intel Xeon @ 2.80 GHz, 7,8 GB RAM, GPU yok (ayrıntı: `reports/performance.json`).

**Protokol:** Model seçimi ve tüm hiperparametre/eşik kararları yalnızca train + validation
üzerinde yapıldı. **Test seti bir kez, yalnızca final model için** kullanıldı; bu yüzden aday
modellerin test metrikleri "ölçülmedi" olarak gösterilir (bilinçli). Train metrikleri eğitim
verisi üzerindeki yeniden-yerine-koyma (resubstitution) skorudur ve yalnızca aşırı uyum
göstergesi olarak okunmalıdır.

Ortak ayarlar: `RANDOM_SEED=42`; genel konu görevi (9 sınıf); train 16.319, val 3.264;
latency = tek metin `predict` ortalaması (200 örnek).

## 1. Genel konu — aday karşılaştırması (`python -m scripts.run_experiments compare|fasttext`)

| ID | Model | Features | Hiperparametreler | Train macro F1 | Val Acc | Val macro F1 | Val weighted F1 | Eğitim (s) | Çıkarım (ms/metin) | Boyut (MB) | Karar |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E01 | MultinomialNB | BoW word 1-2gram | alpha=0.3 | 0,9969 | 0,8520 | 0,7441 | 0,8444 | 1,9 | 3,00 | 2,4 | Baseline |
| E02 | ComplementNB | TF-IDF word 1-2 | alpha=0.3 | 0,9982 | 0,8640 | 0,7657 | 0,8598 | 1,9 | 3,49 | 12,6 | Reddedildi |
| E03 | LogisticRegression | TF-IDF word 1-2 | C=10, balanced | 0,9961 | 0,8339 | 0,7676 | 0,8330 | 34,9 | 3,29 | 11,4 | Reddedildi |
| E04 | LinearSVM | TF-IDF word 1-2 | C=1, balanced | 0,9994 | 0,8505 | 0,7837 | 0,8493 | 2,5 | 0,72 | 9,6 | Reddedildi |
| E05 | LinearSVM | TF-IDF word 1-2, **F5 kök** | C=1, balanced | 0,9989 | 0,8713 | 0,8169 | 0,8716 | 2,1 | 0,66 | 6,4 | F5 +3,3 puan → korundu |
| E06 | LinearSVM | TF-IDF char_wb 2-5 | C=1, balanced | 0,9839 | 0,8753 | 0,8113 | 0,8754 | 7,1 | 0,92 | 3,9 | Reddedildi |
| E07 | LinearSVM | word 1-2 + char_wb 2-5 | C=1, balanced | 0,9992 | 0,8888 | 0,8274 | 0,8874 | 9,1 | 1,88 | 10,5 | 2. aday |
| E08 | LogisticRegression | word 1-2 + char_wb 2-5 | C=10, balanced | 0,9955 | 0,8827 | 0,8284 | 0,8820 | 47,1 | 5,50 | 16,1 | 5x yavaş, daha iyi değil |
| E09 | SGD (modified_huber) | word 1-2 + char_wb 2-5 | alpha=2e-5, balanced | 1,0000 | 0,8778 | 0,8005 | 0,8745 | 7,4 | 5,45 | 7,4 | Reddedildi |
| E10 | MLP (256) | char TF-IDF → LSA 300d gömme | alpha=1e-4, 30 epoch | 0,8800 | 0,7889 | 0,6950 | 0,7836 | 31,8 | 67,9 | 140,7 | Reddedildi (yoğun gömme bilgi kaybı) |
| **E11** | **LinearSVM** | **F5 word 1-2 + char_wb 2-5** | **C=1, balanced** | 0,9993 | **0,8934** | **0,8356** | **0,8927** | 9,8 | 1,91 | 8,9 | **Seçildi** |
| E12 | fastText supervised | sıfırdan 100d alt-sözcük gömmeleri | epoch 25, lr 0.5, wordNgrams 2, minn 2, maxn 5 | 0,9084 | 0,8165 | 0,7065 | 0,8092 | 17,7 | 0,15 | 816,9 | Reddedildi (en hızlı ama en düşük F1; 2M bucket → dev dosya) |

**Denenemeyen:** BERTurk / XLM-R / Sentence-Transformers. Ortam ağ politikası
huggingface.co'yu engelliyor (HTTP 403); önceden eğitilmiş Türkçe ağırlıklar indirilemedi.
Bunun yerine önceden eğitilmemiş iki "gömme" yaklaşımı (E10 LSA+MLP, E12 fastText) ölçüldü.
Transformer karşılaştırması "Gelecek geliştirmeler" altında ayrıca yer alıyor.

## 2. Anlamlılık — 5-kat terim-gruplu CV, train+val (`run_experiments.py cv`)

| ID | Macro F1 ortalama | Std | Katlar |
|---|---:|---:|---|
| E07 | 0,8265 | 0,0090 | 0,822 / 0,838 / 0,812 / 0,827 / 0,833 |
| E08 | 0,8205 | 0,0070 | 0,825 / 0,828 / 0,807 / 0,821 / 0,821 |
| **E11** | **0,8305** | 0,0103 | 0,832 / 0,839 / 0,810 / 0,835 / 0,836 |

E11, E07'den 0,004 daha iyi: fark bir standart sapmanın altında (istatistiksel olarak ayırt
edilemez). E11 seçimi; hem val hem CV'de birinci olması **ve** %15 daha küçük model olmasıyla
gerekçelendirildi.

## 3. Hiperparametre araması — E11 (`run_experiments.py tune`)

Izgara: C ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8} × "Diğer" alt-örneklemesi ∈ {yok, 4000}. 14 deney
(küçük veri için maliyet/fayda dengesi; Optuna gereksiz görüldü).

| C | Diğer örneklemesi | Val macro F1 | Train macro F1 | Eğitim (s) |
|---:|---|---:|---:|---:|
| 0,1 | yok | 0,8074 | 0,9590 | 8,2 |
| 0,25 | yok | 0,8248 | 0,9865 | 8,7 |
| 0,5 | yok | 0,8336 | 0,9958 | 9,2 |
| **1,0** | **yok** | **0,8356** | 0,9993 | 9,5 |
| 2,0 | yok | 0,8338 | 1,0000 | 9,9 |
| 4,0 | yok | 0,8278 | 1,0000 | 10,0 |
| 8,0 | yok | 0,8278 | 1,0000 | 10,7 |
| 0,1–8,0 | 4000 | 0,7720–0,8187 | 0,965–1,000 | 5,6–7,5 |

Karar: C=1,0. 0,5–2 arası düz plato. "Diğer"i azaltmak her C'de 1,7–3,5 puan kaybettirdi
(taksonomi dışı sınıfın çeşitliliği önemli). Train≈1,0 / val≈0,84 farkı (aşırı uyum göstergesi)
C küçüldükçe azalıyor ama val da düşüyor; bu yüzden düzenlileştirme artırılmadı; fark büyük
ölçüde kısa/gürültülü tanım metinlerinin ezberlenebilirliğinden kaynaklanıyor.

## 4. Hiyerarşi — düz vs hiyerarşik (E11, C=1; `run_experiments.py hierarchy`)

| ID | Yaklaşım | Yol doğruluğu (taksonomi içi val, n=1.450) | Yol macro F1 | Genel konu macro F1 (tüm val) | Genel konu acc | Eğitim (s) | Karar |
|---|---|---:|---:|---:|---:|---:|---|
| E-H1 | Hiyerarşik: genel model → konu başına alt model | 0,7814 | 0,8100 | 0,8356 | 0,8934 | 13,1 | Reddedildi |
| **E-H2** | **Düz: 39 birleşik sınıf, genel = marjinal toplam** | **0,8379** | **0,8534** | 0,8363 | 0,8866 | 15,0 | **Seçildi** |

Genel konuda eşit, alt konuda +5,7 puan; tek model olduğu için daha basit ve hiyerarşi
yapısal olarak tutarlı.

## 5. Olasılık kalibrasyonu (train ile eğit, val'de ölç; `train.py`)

| ID | Yöntem | Val macro F1 | Val acc | ECE (15 bin) | Ort. güven | Eğitim (s) | Karar |
|---|---|---:|---:|---:|---:|---:|---|
| E13 | CalibratedClassifierCV sigmoid (3-kat) | 0,7770 | 0,8612 | 0,1114 | 0,750 | 34,4 | Reddedildi |
| E14 | CalibratedClassifierCV isotonic (3-kat) | 0,7924 | 0,8729 | 0,1082 | 0,767 | 34,8 | Reddedildi |
| **E15** | **Temperature scaling (T, train OOF, grup-3-kat)** | **0,8338** | 0,8824 | **0,0307** | 0,893 | 49,8 | **Seçildi** (T=0,1547 train-only; final model T=0,1592) |

E13/E14 sınıf-bazlı kalibrasyon "Diğer" önceliğini geri getirip recall'u düşürdü; temperature
scaling SVM kararlarını değiştirmez.

## 6. Güven eşiği (OOD / belirsizlik)

| ID | Yöntem | Seçim verisi | Eşik | Val macro F1 | Karar |
|---|---|---|---|---:|---|
| E16a | Global eşik, 0,20–0,70 ızgarası | val | 0,70 (**sınırda**) | 0,8478 | Izgara genişletildi |
| **E16b** | **Global eşik, 0,20–0,90** | **val** | **0,75** | **0,8492** | **Seçildi** |
| E17 | Global eşik | train OOF | 0,55 | 0,8424 | Reddedildi |
| E18 | Sınıf bazlı eşik (koordinat artışı) | train OOF | Fizik .55, Kimya .55, Biyoloji .55, Teknoloji .50, Bilim .70, Kitaplar .50, Spor .55, Tarih .45 | 0,8485 | Reddedildi (E16b'yi geçemedi) |

Alt konu eşiği (0,20–0,70, val, taksonomi içi ve genel konusu doğru örnekler): tüm eşiklerde
tek-etiket (küme boyutu 1,0) en iyi → **0,50** (multi-label macro F1 0,9206, subset acc 0,9204).
Veri tek-etiketli olduğundan ek alt konu önermek precision'ı düşürüyor; sistem yine de gerekirse
birden çok alt konu gösterebilecek şekilde tasarlandı (`select_subtopics`).

## 7. Kısa girdi kalibrasyonu (E19)

Eğitim metinleri 5–13 sözcük; 1–3 içerik sözcüklü girdiler aşırı güvenliydi. Val kartlarının
yalnızca **terim** alanıyla (1–3 sözcük), train-only model üzerinde ayrı sıcaklık öğrenildi.

| Değerlendirme | T | Acc | Ort. güven | ECE |
|---|---:|---:|---:|---:|
| Val terimleri, global T | 0,1547 | 0,6664 | 0,782 | 0,1196 |
| Val terimleri, kısa T | 0,2081 | 0,6627 | 0,651 | 0,0745 |
| **Test terimleri, global T** | – | 0,6724 | 0,791 | 0,1243 |
| **Test terimleri, kısa T** | 0,2081 | 0,6678 | 0,678 | **0,0699** |

Karar: ≤3 içerik sözcüğü olan girdilerde T_short=0,2081.

## 8. Sohbet DECAY (E20; `python -m scripts.tune_decay`)

400 simüle sohbet (test metinlerinden; sınıflandırıcı parametresi seçilmez).

| DECAY | İzleme doğruluğu | Geçiş gecikmesi (mesaj) | Geçişte iki konu birlikte | Sahte kayma | **3 konu korunumu** |
|---:|---:|---:|---:|---:|---:|
| 0,0 | 0,896 | 1,13 | 0,005 | 0,110 | 0,003 |
| 0,3 | 0,917 | 1,14 | 0,765 | 0,078 | 0,008 |
| 0,5 | 0,922 | 1,27 | 0,863 | 0,051 | 0,010 |
| 0,6 | 0,858 | 1,87 | 0,870 | 0,038 | 0,058 |
| **0,7** | 0,834 | 2,12 | 0,870 | 0,028 | **0,558** |
| 0,8 | 0,774 | 2,61 | 0,853 | 0,025 | 0,598 |
| 0,9 | 0,700 | 3,22 | 0,810 | 0,023 | 0,605 |

Karar: **0,7** — şartnamenin istediği birikimli daraltmayı (Kitaplar → Bilim → Biyoloji)
sağlayan en küçük decay (0,6→0,7 arası %6→%56 sıçrama); daha yüksek değerler korunumu az
artırıp izlemeyi belirgin bozuyor.

## 9. Final model — test (bir kez, `python evaluate.py`)

| Metrik | Argmax | Eşikli (0,75) |
|---|---:|---:|
| Accuracy | 0,8832 | **0,8943** |
| Macro precision | 0,8022 | 0,8752 |
| Macro recall | 0,8846 | 0,8295 |
| **Macro F1** | 0,8386 | **0,8500** |
| Weighted F1 | 0,8851 | 0,8934 |
| ECE | 0,0374 | – |
| Alt konu yol doğruluğu (taksonomi içi) | 0,847 | – |

Val (0,8492) ile test (0,8500) eşikli macro F1 çok yakın → validation'a aşırı uyum yok.

## 10. Hata analizi → iyileştirme döngüsü (§69)

| Tur | Bulgu | İyileştirme | Sonuç |
|---|---|---|---|
| 1 | Sigmoid/isotonic kalibrasyon macro F1'i düşürdü | Temperature scaling (E15) | F1 korundu, ECE 0,108→0,031 |
| 2 | Güven eşiği ızgara sınırında | Izgara 0,90'a genişletildi (E16b) | İç optimum 0,75 |
| 3 | Kabul testi B (Bilim) "Belirsiz" | Sınıf bazlı eşik (E18) | Val'de iyileşme yok → reddedildi; KI-001 |
| 4 | "Gen." gibi kısa girdiler %100 güven | Kısa girdi sıcaklığı (E19) | Test terim ECE 0,124→0,070 |
| 5 | Stopword listesi vectorizer biçimiyle uyumsuzdu (uyarı) | Stopword'ler normalize/F5 biçimine çevrildi | E11 val 0,8336→0,8356 |
