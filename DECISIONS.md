# Karar Kaydı (ADR)

## ADR-001 — Birincil veri: Türkçe Tabu Veri Seti
**Decision:** Eğitim verisi `wleeaf/taboo-dataset` (MIT, 37.278 kart).
**Reason:** 8 hedef konunun 41 kaynak kategoriyle alt konu düzeyinde eşlenebilmesi; kuantum
bilgisayar terimleri; doğal "Diğer" ve OOD sınıfı; açık lisans; sabit commit ile tekrar üretim.
**Alternatives:** Haber setleri (TTC-4900, Interpress, TRT, Milliyet), TurkishMMLU, TQuAD.
**Result:** 29.585 temiz örnek; DATASET_RESEARCH.md.

## ADR-002 — Kaggle/UCI/HF yerine GitHub kaynakları
**Decision:** Yalnızca sabit commit'li GitHub ham dosyaları, dosya bazında SHA-256.
**Reason:** Ortam ağ politikası Kaggle/UCI/HF'yi engelliyor; GitHub arşiv (codeload) de
engelli olduğundan tar yerine 150 dosya tek tek indirilip doğrulanıyor.
**Result:** `scripts/download_data.py`; bayt düzeyinde tekrar üretilebilir veri.

## ADR-003 — Terim-gruplu bölme ve görülmemiş OOD kategorileri
**Decision:** StratifiedGroupKFold (grup = terim sözcüklerinin ilk 6 harfi); "Diğer"in %35'i
eğitimden tamamen çıkarıldı.
**Reason:** Aynı terimin çekim/varyasyonlarıyla sızıntıyı engellemek; OOD'yi gerçekten görülmemiş
alanlarla ölçmek.
**Result:** Grup/metin/yakın-kopya çakışması 0.

## ADR-004 — Dışlanan 31 komşu kategori
**Decision:** Anlamca taksonomiyle örtüşen kategoriler hiçbir bölmede kullanılmıyor.
**Alternatives:** "Diğer" yapmak (sistematik etiket gürültüsü) veya pozitif sınıfa katmak
(taksonomiyi keyfi genişletmek).
**Result:** 7.693 kart dışlandı; TAXONOMY.md.

## ADR-005 — TF-IDF baseline → F5 word + char TF-IDF + LinearSVM
**Decision:** E11 (val macro F1 0,836; CV 0,831 ± 0,010).
**Reason:** En iyi val ve CV skoru; E07 ile fark < 1σ olduğundan daha küçük model belirleyici.
**Alternatives:** NB, LR, SGD, LSA+MLP, fastText (EXPERIMENTS.md §1).
**Result:** Test macro F1 0,850 (eşikli).

## ADR-006 — Transformer denenmedi
**Decision:** BERTurk/XLM-R/Sentence-Transformers karşılaştırması yapılmadı.
**Reason:** huggingface.co ortam politikasınca engelli (HTTP 403); ölçülmeyen bir modelin daha iyi
olacağı varsayılmadı. Mevcut model CPU'da 2,2 ms/metin ve 22,6 MB.
**Result:** "Gelecek geliştirmeler"de ilk madde; KNOWN_ISSUES KI-004.

## ADR-007 — Düz "Genel > Alt" sınıflandırma (hiyerarşik yerine)
**Decision:** 39 birleşik sınıflı tek model; genel olasılık = marjinal toplam.
**Reason:** Genel konuda eşit (0,836 ile 0,836), alt konu yolunda +5,7 puan (0,838 ile 0,781);
tek model; hiyerarşi yapısal olarak tutarlı.

## ADR-008 — Temperature scaling (sigmoid/isotonic yerine)
**Decision:** Tek parametreli sıcaklık, train içi grup-OOF NLL ile.
**Reason:** Sigmoid/isotonic macro F1'i 0,777 / 0,792'ye düşürdü; temperature 0,834'ü korudu ve
ECE 0,031 verdi.

## ADR-009 — Güven eşiği 0,75, sınıf bazlı eşik yok, kısa girdi sıcaklığı
**Decision:** Global eşik validation'da (ızgara 0,20–0,90) 0,75; sınıf bazlı OOF eşikleri val'de
daha kötü olduğu için reddedildi; ≤3 içerik sözcüklü girdilerde T_short=0,2081.
**Result:** Test eşikli macro F1 0,850; OOD reddetme %88,2; kısa girdi ECE 0,124→0,070.

## ADR-010 — DECAY = 0,7 (deneyle doğrulandı)
**Decision:** Başlangıç değeri korundu.
**Reason:** Birikimli üç konu korunumu 0,6'da %6 → 0,7'de %56; 0,8+ izlemeyi bozuyor
(EXPERIMENTS.md §8).

## ADR-011 — Rol tabanlı tema birleştirme
**Decision:** Sabit ikili tablo yerine rol (ortam / niteleyici / alan) dilbilgisi.
**Reason:** 8 konunun tüm kombinasyonlarına genellenir (28 ikili kural yerine 8 konu tanımı);
"bilimsel kitaplar", "tarih kitapları", "biyoloji hakkında bilimsel kitaplar" üretir.
**Limitations:** Konu adlarının Türkçe biçimleri elle tanımlı; yeni bir konu eklemek için
`TOPIC_FORMS`'a bir satır gerekir.

## ADR-012 — Model ağırlıklarından anahtar sözcüklü sorgu
**Decision:** Sorgu = tema çekirdeği + son mesajdan en yüksek pozitif SVM ağırlıklı ≤2 sözcük
(ağırlık ≥ 0,30 ≈ sınıf başına pozitif ağırlıkların 90. yüzdeliği), fiil filtresi, hal eki kırpma.
**Reason:** "kübit", "penaltı" gibi konuyu belirleyen sözcükleri seçer; "kullanır", "dün" gibi
sözcükleri eler. Harici bağımlılık (morfolojik analizör) gerektirmez.

## ADR-013 — Wikipedia birincil, DuckDuckGo yedek; devre kesici
**Decision:** API anahtarsız iki kaynak; bağlantı yoksa 120 sn askıya alma; SQLite önbellek 24 sa.
**Reason:** Çevrimdışıyken her mesajda ~6,1 sn beklemeyi 8 ms'ye indiriyor.

## ADR-014 — Güvenli model yükleme
**Decision:** joblib artifact'ı yalnızca metadata'daki SHA-256 eşleşirse yüklenir; sınıf listesi
doğrulanır; yazma atomiktir.
**Reason:** pickle tabanlı format güvenilmeyen dosyada kod çalıştırabilir.

## ADR-015 — Sentetik veri ve augmentation yok
**Decision:** Kullanılmadı.
**Reason:** Kabul testi cümlelerine benzer sentetik veri test sızıntısı riski; veri boyutu (16 bin)
ve sınıf başına ≥178 örnek, basit doğrusal model için yeterli; augmentation için güvenilir Türkçe
paraphrase/back-translation modeli ortamda yok.
