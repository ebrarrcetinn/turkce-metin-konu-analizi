# Veri Seti Araştırması

Tarih: 2026-09-24/25. Hedef: Fizik, Kimya, Biyoloji, Teknoloji, Bilim, Kitaplar, Spor, Tarih
konularını ve alt konularını (örn. Kuantum Mekaniği ↔ Kuantum Bilgisayarlar) ayırabilen, açık
lisanslı Türkçe metin verisi.

## Erişim kısıtı (önemli)

Çalışma ortamının ağ politikası **Kaggle, UCI, Hugging Face ve tr.wikipedia.org erişimini
engelliyor** (HTTP 403, `reports/` altında değil, kurulum günlüğünde gözlendi). Açık olan
kaynaklar: GitHub (`raw.githubusercontent.com`, `git clone`) ve PyPI. Bu nedenle:

* Kaggle/UCI/HF adayları **indirilmeden**, yayımlanmış dokümantasyonları (README, veri kartı,
  makale) üzerinden değerlendirildi; bu adaylar için satır/sınıf bilgisi kaynağın beyanıdır,
  tarafımızca doğrulanmamıştır.
* GitHub üzerinden erişilebilen adaylar klonlanıp doğrudan incelendi (✔ ile işaretli).

## Aday tablosu

| Dataset | Kaynak | Satır | Sınıf | Dil | Lisans | Avantaj | Problem | Karar |
|---|---|---:|---:|---|---|---|---|---|
| **Türkçe Tabu Veri Seti** ✔ | GitHub `wleeaf/taboo-dataset` @ `f621b46` | 37.278 | 150 kategori | tr | MIT | 150 ince kategori (kuantum, genetik, osmanlı, futbol, edebiyat…); taksonomiye eşlenebilir; alt konu üretir; kuantum bilgisayar terimleri var | Kısa (5–13 sözcük) tanım metinleri; %28 terim–açıklama uyuşmazlığı ve %27 kesik açıklama (60 kartlık elle inceleme, bkz. DATASET_CARD); büyük olasılıkla kısmen makine üretimi | **Birincil veri** |
| Turkish BQuAD ✔ | GitHub `TurQuest/turkish-bquad` @ `30a3e07` | 154 belge / MEB lise biyoloji paragrafları | tek alan | tr | MIT | İnsan yazımı, ders kitabı üslubu | Tek konu; sınıflandırma verisi değil | **Harici değerlendirme** (Biyoloji) |
| Osmanlı Tarihi RC ✔ | GitHub `okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset` @ `c6852f8` | 11+ belge (2020-enelpi alt kümesi) | tek alan | tr | MIT | İnsan yazımı, ansiklopedik tarih metni | Tek konu | **Harici değerlendirme** (Tarih) |
| TQuAD ✔ | GitHub `TQuad/turkish-nlp-qa-dataset` | 681 başlık | tek alan (Türk-İslam bilim tarihi) | tr | **Belirtilmemiş** | Bilim tarihi içeriği | Lisans yok → kullanılamaz | Reddedildi |
| kuzgnlar QA ✔ | GitHub `kuzgnlar/datasets` | 129 başlık | karışık | tr | GPL-3.0 | Bilim/yöntem başlıkları var | Çok küçük; GPL yükümlülükleri model/proje lisansını etkiler | Reddedildi |
| TTC-4900 | Kaggle / HF `savasy/ttc4900` | 4.900 (beyan) | 7 (dünya, ekonomi, kültür, sağlık, siyaset, spor, teknoloji) | tr | Doğrulanamadı | Temiz haber verisi | Taksonomiyle yalnızca Spor/Teknoloji örtüşüyor; ortamdan erişilemedi | Reddedildi |
| TC32 | Kaggle `savasy/multiclass-classification-data-for-turkish-tc32` | ~430 bin (beyan) | 32 ürün kategorisi | tr | Doğrulanamadı | Büyük | Ürün şikâyet/yorumları; konu taksonomisiyle ilgisiz | Reddedildi |
| Interpress News (270K lite) | HF `interpress_news_category_tr_lite` | ~273 bin (beyan) | 10 haber kategorisi | tr | Doğrulanamadı | Büyük, haber | Bilim alt dallarını (fizik/kimya/biyoloji) ayırmıyor; ortamdan erişilemedi | Reddedildi |
| TRT 11 kategori | GitHub `gurkan08/datasets` | beyan edilmemiş | 11 (bilim_teknoloji birleşik) | tr | Belirtilmemiş | Haber | Bilim ve teknoloji tek sınıf → kuantum ayrımı imkânsız; RAR arşivi | Reddedildi |
| Milliyet Haber Derlemi | GitHub `ogozuacik/turkce-haber-derlemi` | 116.068 (beyan) | ~199 ham kategori | tr | MIT | Büyük haber derlemi | Veri Git LFS'de; LFS indirme bağlantısı ortam politikasınca kısıtlı; haber kategorileri bilim alt dallarını ayırmıyor | Reddedildi |
| TurkishMMLU | HF `AYueksel/TurkishMMLU` | ~10 bin soru | 9 ders (fizik, kimya, biyoloji, tarih…) | tr | Erişim e-postayla (kontrollü) | Taksonomiye en yakın konu yapısı | Kontrollü erişim; kıyas (benchmark) verisi olarak eğitimde kullanımı uygun değil | Reddedildi |
| UCI — Türkiye Öğrenci Değerlendirme | UCI | — | — | sayısal | — | — | Metin değil (anket puanları); UCI'da Türkçe konu sınıflandırma verisi bulunamadı | Uygun değil |

## Seçim gerekçesi

1. **Taksonomi uyumu:** Tabu veri setinin 150 kategorisinden 41'i hedef 8 genel konunun
   alt konularına doğrudan eşleniyor (ör. `kuantum`→Fizik > Kuantum Mekaniği, `genetik`→Biyoloji >
   Genetik, `osmanli`→Tarih > Osmanlı Tarihi). Haber veri setlerinin hiçbiri Fizik/Kimya/Biyoloji
   ayrımını ve alt konu düzeyini sağlamıyor.
2. **Kritik kuantum ayrımı:** Kaynakta kübit, kuantum kapısı, kuantum işlemcisi gibi kartlar
   bulunuyor; bunlar şeffaf bir terim kuralıyla Teknoloji > Kuantum Bilgisayarlar'a ayrıldı.
3. **Negatif (taksonomi dışı) sınıf:** Kalan 78 kategori ("gastronomi", "hukuk", "moda"…) doğal
   bir "Diğer" sınıfı ve eğitimde hiç görülmeyen 27 kategoriyle gerçek bir OOD testi sağlıyor.
4. **Lisans ve provenance:** MIT; sabit commit ve dosya bazında SHA-256 ile tekrar üretilebilir.
5. **Dengeli kategori büyüklüğü:** Kategori başına 237–250 kart.

## Kabul edilen zayıflıklar

* Metinler kısa tanım cümleleri; gerçek sohbet mesajlarından üslup olarak farklı. Bu farkı
  ölçmek için insan yazımı iki harici set (BQuAD, Osmanlı RC) ayrı değerlendirildi
  (FINAL_REPORT §14).
* "Bilim" sınıfının tek kaynağı `epistemoloji` kategorisi; "bilimsel yöntem" cümleleri için
  zayıf (KNOWN_ISSUES KI-001).
* Sentetik veri **kullanılmadı** (kabul testi cümlelerine yakın sentetik veri üretmek test
  sızıntısı riski taşıyacağı için bilinçli olarak kaçınıldı).
