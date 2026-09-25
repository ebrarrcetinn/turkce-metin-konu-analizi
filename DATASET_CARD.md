# Dataset Card — Türkçe Konu Sınıflandırma Verisi (split-v1)

## Kaynaklar ve provenance

| Anahtar | Veri seti | URL | Commit | Lisans | Dosya | Bayt | Rol |
|---|---|---|---|---|---:|---:|---|
| `taboo` | Türkçe Tabu Veri Seti | https://github.com/wleeaf/taboo-dataset | `f621b460f4511f2afb1a89f87b2d94af9440b09c` | MIT | 150 JSON | 12.499.080 | eğitim / validation / test / OOD |
| `bquad` | Turkish BQuAD | https://github.com/TurQuest/turkish-bquad | `30a3e070e5eeb3ba04da89cac3a426eb858df032` | MIT | 2 JSON | 758.468 | harici değerlendirme (Biyoloji) |
| `ottoman` | Osmanlı Tarihi Okuduğunu Anlama | https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset | `c6852f86f25e093ead13d8e2d0d87d92deadf727` | MIT | 6 JSON | 3.245.435 | harici değerlendirme (Tarih) |

* İndirme: `python -m scripts.download_data` — her dosya sabit commit'ten indirilir ve
  `src/data/taboo_manifest.json` / `src/data/sources.py` içindeki SHA-256 ile doğrulanır;
  uyuşmazlıkta dosya silinir ve hata verilir. İndirme tarihi ve özetler
  `data/raw/provenance.json` dosyasına yazılır (ilk indirme: 2026-09-24 UTC).
* Ham veri repoya commit edilmez (`.gitignore`: `data/raw/`, `data/processed/`).

## Kolonlar

Ham tabu kartı: `id`, `kategori`, `kelime` (terim), `aciklama` (tanım), `yasakli_kelimeler`,
`zorluk` (kolay 17.815 / orta 14.358 / zor 5.105). Kullanılan alanlar: `kategori`, `kelime`,
`aciklama`. İşlenmiş satır: `text` (= terim + " " + tanım), `general`, `subtopic`, `category`,
`term`, `group`, `source`.

## Etiketler

* 9 genel etiket: Fizik, Kimya, Biyoloji, Teknoloji, Bilim, Kitaplar, Spor, Tarih, Diğer.
* 38 alt konu; model sınıfları 39 birleşik etiket ("Genel > Alt" + "Diğer"). Eşleme: TAXONOMY.md.

## Temizleme, dışlama ve dönüşümler

| Adım | Sonuç |
|---|---:|
| Ham kart | 37.278 |
| Boş tanım | 0 |
| Taksonomiyle anlamsal çakışan 31 kategori dışlandı (TAXONOMY.md) | −7.693 |
| Birebir aynı (normalize) metin + etiket | 0 |
| Aynı metin, farklı etiket (çelişki) | 0 |
| **Temizlenmiş** | **29.585** |

Dönüşümler: terim+tanım birleştirme; `kuantum` kategorisindeki 32 kuantum-bilgisayar kartının
terim kuralıyla Teknoloji'ye taşınması; ön işleme (normalize) model hattında uygulanır, veride
kalıcı değişiklik yapılmaz.

## Dağılım

| Bölme | Satır | Diğer | Spor | Biyoloji | Teknoloji | Tarih | Fizik | Kitaplar | Kimya | Bilim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 16.319 | 9.053 | 1.607 | 1.418 | 1.265 | 885 | 851 | 710 | 352 | 178 |
| val | 3.264 | 1.814 | 320 | 283 | 252 | 178 | 170 | 141 | 71 | 35 |
| test | 3.263 | 1.812 | 320 | 285 | 249 | 175 | 174 | 141 | 71 | 36 |
| ood_unseen | 6.060 | 6.060 | – | – | – | – | – | – | – | – |
| external | 800 | – | – | 400 | – | 400 | – | – | – | – |

* Dengesizlik: taksonomi içinde en büyük/en küçük sınıf oranı **9,02** (Spor / Bilim);
  "Diğer" etiketli verinin %55,5'i. Önlem: `class_weight="balanced"`; test'e örnekleme yok.
  "Diğer" alt-örneklemesi denendi ve validation macro F1'i düşürdü (EXPERIMENTS.md E11-tune).
* En küçük alt konu: Kuantum Bilgisayarlar (32 kart).

## Bölme stratejisi

* 70/15/15; **terim-gruplu** (`StratifiedGroupKFold`, grup = her sözcüğün ilk 6 harfi) ve
  kaynak kategoriye göre stratified; `RANDOM_SEED=42`.
* "Diğer" kategorilerinin %35'i (27 kategori) eğitimden tamamen çıkarılıp `ood_unseen` yapıldı;
  eğitimdeki bir terimle aynı grupta olan OOD kartları atıldı.
* Sızıntı ölçümü (`data/processed/split_report.json`): train–val ve train–test grup çakışması 0;
  birebir metin çakışması 0; char 3–5gram TF-IDF kosinüs ≥ 0,9 yakın-kopya oranı val/test/harici
  için 0,0.

## EDA özeti (`reports/eda.json`, `reports/figures/`)

* Metin uzunluğu: 5–13 sözcük (ortalama 10,7); harici setler 6–38 sözcük (ortalama 13,8).
* Sözlük: 48.307 yüzey biçimi; tek kez geçenlerin oranı %52,3 (Türkçe eklemeli morfoloji → karakter
  n-gram ve F5 kök kullanımının gerekçesi).
* Anomaliler (ham tanımlarda): URL 0, emoji 0, mention 0, hashtag 0, mojibake 0; HTML benzeri 1
  (aslında Dirac notasyonu `<bra| |ket>`, yanlış pozitif); alışılmadık karakter içeren 133;
  sezgisel "kesik ek" kalıbı 977.
* Tekrarlar: terim+tanım birebir tekrar 0; farklı kategorilerde aynı terim 2.540; aynı tanım 48.

## Veri kalitesi — elle inceleme

Rastgele 60 kart (`sample(60, random_state=42)`) tek tek okunarak elle incelendi (tek
değerlendirici; örneklem küçük olduğu için oranlar yaklaşıktır):

| Bulgu | Oran | Örnek |
|---|---:|---|
| Terim ile tanım uyuşmuyor (tanım başka bir kavramı anlatıyor) | 17/60 (~%28) | `kefir :: farklı ağırlık disklerini üzerinde toplayan dikey istasyon`; `özofagus :: kolun arka yüzünde uzanan… üç başlı kas` |
| …bunların yanlış tanımı yine **aynı kategoriden** | 17/17 | `arkeoloji`: `krater :: …dairesel dizilmiş dev taş bloklar (Stonehenge)` |
| Tanımda eksik sözcük / kesik cümle | 16/60 (~%27) | `ductus :: …kalemin veya ve hızı.` |
| Kategori düzeyinde açık etiket hatası | 0/60 | – |

Sonuç: terim–tanım gürültüsü yüksek, ancak **kategori etiketi** — bu projenin öğrendiği sinyal —
incelenen örneklemde tutarlı. Uyuşmayan kartlar modele aynı kategoriden iki kavramı birlikte
gösterdiği için kategori düzeyinde zararı sınırlıdır; yine de hata analizinde görülen bazı
"hatalar" aslında etiket gürültüsüdür (ERROR_ANALYSIS.md §5).

## Bilinen bias ve sınırlamalar

* Üslup: ansiklopedik/sözlük tanımları; günlük sohbet dili, ironi, argo yok.
* Kısmen makine üretimi izlenimi (tek tip uzunluk, kalıp ifadeler, kesik cümleler).
* Kültürel odak: Türkiye (Osmanlı, Cumhuriyet, Türk halk edebiyatı) — bilinçli, ama "Tarih"
  sınıfı dünya tarihinin geri kalanında daha zayıf olabilir.
* "Bilim" sınıfı yalnızca epistemoloji; "Kitaplar" sınıfı edebiyat terimleri ağırlıklı (belirli
  kitap/yazar adları az).
* Harici setler tek konuludur; yalnızca recall ölçülebilir.
