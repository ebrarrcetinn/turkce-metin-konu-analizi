# Model Karşılaştırma Raporu

Kaynak: `reports/experiments.json` (validation), `reports/evaluation.json` (test, yalnızca final).
Ayrıntılı deney kayıtları: EXPERIMENTS.md.

| Model | Val Accuracy | Val Macro F1 | Val Weighted F1 | 5-kat CV Macro F1 | Inference (ms/metin) | Boyut (MB) | Karar |
|---|---:|---:|---:|---:|---:|---:|---|
| BoW + MultinomialNB (E01) | 0,852 | 0,744 | 0,844 | – | 3,0 | 2,4 | Baseline |
| TF-IDF + ComplementNB (E02) | 0,864 | 0,766 | 0,860 | – | 3,5 | 12,6 | – |
| TF-IDF + LogisticRegression (E03) | 0,834 | 0,768 | 0,833 | – | 3,3 | 11,4 | – |
| TF-IDF + LinearSVM (E04) | 0,851 | 0,784 | 0,849 | – | 0,7 | 9,6 | – |
| TF-IDF(F5) + LinearSVM (E05) | 0,871 | 0,817 | 0,872 | – | 0,7 | 6,4 | – |
| char n-gram + LinearSVM (E06) | 0,875 | 0,811 | 0,875 | – | 0,9 | 3,9 | – |
| word+char + LinearSVM (E07) | 0,889 | 0,827 | 0,887 | 0,827 ± 0,009 | 1,9 | 10,5 | 2. |
| word+char + LogisticRegression (E08) | 0,883 | 0,828 | 0,882 | 0,821 ± 0,007 | 5,5 | 16,1 | – |
| word+char + SGD (E09) | 0,878 | 0,801 | 0,875 | – | 5,4 | 7,4 | – |
| LSA-300 gömme + MLP (E10) | 0,789 | 0,695 | 0,784 | – | 67,9 | 140,7 | – |
| fastText supervised (E12) | 0,817 | 0,707 | 0,809 | – | 0,2 | 816,9 | – |
| **F5 word + char + LinearSVM (E11)** | **0,893** | **0,836** | **0,893** | **0,831 ± 0,010** | 1,9 | 8,9 | **Seçildi** |

## Final sistem (E11 + düz hiyerarşi + temperature scaling + güven eşiği 0,75)

| Set | Accuracy | Macro F1 | Weighted F1 | Not |
|---|---:|---:|---:|---|
| Validation (train-only model, eşikli) | 0,897 | 0,849 | 0,896 | eşik burada seçildi |
| **Test (eşikli, bir kez)** | **0,894** | **0,850** | **0,893** | ECE 0,037 |
| Test (argmax) | 0,883 | 0,839 | 0,885 | |
| OOD (27 görülmemiş kategori, 6.060 metin) | reddetme oranı **0,882** | – | – | yanlış kabul 0,118 |
| Harici BQuAD (Biyoloji, 400) | recall 0,730 (argmax) / 0,573 (eşikli) | – | – | alan/üslup kayması |
| Harici Osmanlı (Tarih, 400) | recall 0,660 (argmax) / 0,445 (eşikli) | – | – | alan/üslup kayması |

Final çalışma zamanı modeli: 22,6 MB, yükleme 0,93 s, tek metin p50 2,2 ms / p95 2,5 ms
(2 vCPU, GPU yok).

## Seçim gerekçesi (ölçülebilir)

1. **En yüksek validation macro F1** (0,836) ve **en yüksek CV ortalaması** (0,831).
2. E07'ye göre farkı istatistiksel olarak ayırt edilemez (Δ=0,004 < 1σ); bu durumda **daha küçük
   model** (8,9 vs 10,5 MB) belirleyici oldu.
3. LogisticRegression (E08) doğal olasılık verse de 5× yavaş eğitim, 3× yavaş çıkarım ve daha
   düşük CV F1; olasılıklar temperature scaling ile SVM'e eklendi (ECE 0,031).
4. Yoğun gömme yaklaşımları (E10, E12) bu kısa ve morfolojik olarak zengin metinlerde
   seyrek n-gram modellerinin 13–14 puan gerisinde.
5. Transformer ölçülemedi (ağ kısıtı). Ölçülmeyen bir modelin daha iyi olacağı varsayılmadı;
   CPU'da ~2 ms/metinlik mevcut gecikme ve 22 MB boyut, konuşma arayüzü için yeterli.
