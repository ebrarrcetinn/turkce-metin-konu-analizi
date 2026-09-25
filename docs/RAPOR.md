# Türkçe Metin Konu Analizi — Proje Raporu

Yazar: Ebrar Cemre Çetin
Tarih: 27.09.2026

## 1. Amaç

Kullanıcının yazdığı Türkçe metnin genel konusunu ve alt konularını bulmak, birden fazla
mesajdan oluşan bir sohbetin genel konusunu takip etmek, bu konu için internette arama
yapmak ve metinleri, konuları, sorguları ve arama sonuçlarını veritabanına kaydetmek.
Ödevde istendiği gibi sınıflandırma algoritmaları hazır makine öğrenmesi kütüphaneleri
yerine proje içinde yazılmıştır.

## 2. Sistem tasarımı

Program nesne yönelimli olarak katmanlara ayrılmıştır. Her katman bir sonrakine yalnızca
kendi sınıfının arayüzüyle bağlanır.

| Katman | Sınıflar | Görev |
|---|---|---|
| Konsol | `ConsoleApp`, `ModelPreparer` (`proje.py`) | Döngü, komutlar, çıktı biçimi; model yoksa veri indirme ve eğitim |
| Uygulama servisi | `ChatService` | Bir mesaj için tüm adımları sırayla çalıştırır, hataları yalıtır |
| Model | `TopicModel`, `Trainer`, `Evaluator` | Tahmin, eğitim, ölçüm |
| Algoritmalar | `TfidfVectorizer`, `CombinedVectorizer`, `MultinomialNaiveBayes`, `SoftmaxRegression`, `ClassificationReport` | Kendi yazılan ML bileşenleri |
| Veri | `DatasetBuilder`, `DataSource` | İndirme, doğrulama, temizleme, bölme |
| Sohbet | `ConversationTracker`, `Theme` | Sohbet konusu ve tema ifadesi |
| Arama | `WebSearchClient` | Wikipedia ve DuckDuckGo |
| Kayıt | `Database` | SQLite tabloları, şema sürümü, önbellek |

Bir mesajın işlenişi:

1. Metin ön işlenir ve vektöre çevrilir.
2. Model 39 sınıfın ("Genel > Alt" ve "Diğer") olasılığını üretir.
3. Genel konu olasılığı, o konunun alt konu olasılıklarının toplamıdır. En olası genel
   konunun olasılığı güven eşiğinin altındaysa sonuç "Belirsiz", en olası sınıf "Diğer" ise
   "taksonomi dışında" olur.
4. Alt konular, seçilen genel konu içindeki koşullu olasılığa göre sıralanır. En olası alt
   konu her zaman, diğerleri payı %15'i aşarsa (en çok 3) listelenir. Ana konu dışında %20'yi
   aşan genel konular "İlişkili Konular" olarak gösterilir.
5. Sohbet skorları güncellenir, tema ifadesi ve arama sorgusu üretilir.
6. Kayıt ve arama yapılır. İnternet veya veritabanı hatası sınıflandırmayı durdurmaz.

## 3. Metin ön işleme

* Python'un `str.lower()` fonksiyonu Türkçe "I" harfini "i", "İ" harfini "i̇" yaptığı için
  özel `turkish_lower` fonksiyonu yazıldı.
* Normalizasyon: bozuk karakter kodlaması onarımı, Unicode NFC, HTML/URL/@kullanıcı
  temizliği, emoji ve rakamların atılması, kesme işaretinden sonraki eklerin atılması
  ("Ankara'da" → "ankara"), noktalama yerine boşluk.
* F5 kökleme: her sözcüğün ilk 5 harfi alınır. Türkçe gibi eklemeli bir dilde basit ama
  etkili bir kökleme yöntemidir ("hücreler", "hücrenin" → "hücre").
* Stopword listesi sözcük özelliklerinden çıkarılır.

## 4. Veri

Birincil kaynak Türkçe Tabu Veri Setidir (MIT, 150 kategori, 37.278 kart). Her kart bir
kavram ve kısa açıklamasından oluşur; eğitim metni "kavram + açıklama"dır.

| Adım | Sayı |
|---|---|
| Ham kart | 37.278 |
| Birden fazla konuya girebilen kategoriler (31 kategori) nedeniyle çıkarılan | 7.693 |
| Temizlik sonrası | 29.585 |
| Eğitim / doğrulama / test | 16.023 / 3.458 / 3.479 |
| Görülmemiş konu – eşik seçimi (`ood_val`) / son ölçüm (`ood_unseen`) | 2.785 / 3.275 |
| Harici değerlendirme (biyoloji + Osmanlı tarihi paragraf cümleleri) | 800 |

Kararlar:

* **Terim gruplu bölme.** Aynı kavramın (ilk 6 harfi aynı sözcükler) kartları hep aynı
  bölmeye düşer. Eğitim ile doğrulama/test arasında ortak grup ve birebir aynı metin sayısı
  0'dır; karakter n-gram kosinüs benzerliği 0,9'u aşan yakın kopya oranı testte %0,06'dır.
* **Görülmemiş konular.** "Diğer" kategorilerinin %35'i eğitimden tamamen ayrıldı. Bunların
  yarısı güven eşiğini seçmekte, diğer yarısı yalnızca son ölçümde kullanıldı; iki yarı ortak
  kategori içermez. Böylece "modelin tanımadığı bir konuda emin olmaması" ölçülebilir hale
  geldi.
* **Konu adı örnekleri.** Tabu oyununda kartın kendi kategori adı yasaklı sözcüktür; bu yüzden
  "biyoloji" sözcüğü biyoloji kartlarında neredeyse hiç geçmez. İlk denemede model tek başına
  "biyoloji" girdisini Teknoloji > Nanoteknoloji olarak sınıflandırdı ("-oloji" karakter
  n-gram'ları nedeniyle). Her alt konu için alt konu adı, genel konu adı ve ikisinin birleşimi
  eğitim setine eklendi (114 örnek). Bu örnekler yalnızca eğitim setindedir ve test sonuçlarını
  etkilemez.

## 5. Algoritmalar

### 5.1 TF-IDF

İki vektörleştirici birleştirilir: F5 kökleri üzerinde sözcük 1-2 gram ve her sözcüğün başına
ve sonuna boşluk eklenmiş karakter 2-5 gram (en az 2 belgede geçenler). Toplam 196.785 özellik.

* IDF: `idf(t) = ln((1 + n) / (1 + df(t))) + 1` (yumuşatılmış; hiçbir terim 0 almaz)
* TF: `1 + ln(tf)` (alt doğrusal; tekrar eden sözcüğün etkisi sınırlanır)
* Her vektör L2 normuyla 1 uzunluğa getirilir.

Sözlük ve IDF yalnızca eğitim verisinden öğrenilir.

### 5.2 Çok terimli Naive Bayes (temel model)

`log P(k | x) ∝ log P(k) + Σ_j x_j · log P(j | k)`, `P(j | k) = (N_kj + α) / (N_k + α·F)`.
200 bine yakın özellikte büyük α tüm sınıfları birbirine benzettiği için α = 0,001 seçildi;
sınıf dengesizliğinin etkisini azaltmak için eşit önsel olasılık kullanıldı.

### 5.3 Softmax regresyon (ana model)

`p = softmax(xW + b)`. Kayıp, sınıf ağırlıklı çapraz entropi ve L2 düzenlileştirmedir. Sınıf
ağırlığı `n / (K · n_k)` biçimindedir; böylece az örnekli sınıflar (Bilim, 182 örnek)
bastırılmaz. Logitlere göre gradyan `(p − y)` olduğundan gradyan
`Xᵀ · (ağırlık · (p − y)) + λW` ile hesaplanır.

* Eniyileme: mini-batch (256) Adam, öğrenme oranı 0,05.
* Seyrek girdide yalnızca o batch'te geçen özelliklerin ağırlıkları güncellenir; bu, 200 bin
  × 39'luk matrisin tamamını her adımda dolaşmaktan çok daha hızlıdır.
* Erken durdurma: her epoch sonunda doğrulama macro F1 ölçülür, 3 epoch iyileşme olmazsa
  durulur ve en iyi epoch'un ağırlıkları alınır.

### 5.4 Kalibrasyon

Temperature scaling: logitler `T`'ye bölünür, `T` doğrulama setindeki negatif
log-olabilirliği en küçükleyecek şekilde altın oran aramasıyla bulunur. Sınıf sıralaması
değişmez, yalnızca güven skorları gerçek doğruluğa yaklaşır.

### 5.5 Güven eşiği

0,20 ile 0,90 arasındaki eşikler, doğrulama seti ile `ood_val` birlikte kullanılarak
denenir. Eşiğin altında kalan tahmin "Diğer" sayılır ve 9 genel konu üzerinden macro F1
hesaplanır. En iyi eşik 0,60 çıktı (macro F1 0,782; 0,20'de 0,752, 0,90'da 0,739).

### 5.6 Metrikler

Doğruluk, sınıf bazında precision / recall / F1, macro ve ağırlıklı F1, karmaşıklık matrisi
ve 15 aralıklı beklenen kalibrasyon hatası (ECE) `src/ml/metrics.py` içinde hesaplanır.

## 6. Eğitim süreci ve model seçimi

| Aşama | Sonuç |
|---|---|
| Naive Bayes, doğrulama macro F1 | 0,833 |
| Softmax regresyon, doğrulama macro F1 | 0,854 (en iyi epoch: 9) |
| Seçilen model | Softmax regresyon |
| Sıcaklık (T) | 0,717 |
| Güven eşiği | 0,60 |
| Son eğitim | Eğitim + doğrulama verisi, 9 epoch |
| Toplam süre (2 çekirdekli CPU) | yaklaşık 55 saniye |

Softmax regresyonun epoch'lara göre doğrulama macro F1 değerleri: 0,710, 0,824, 0,842, 0,848,
0,848, 0,850, 0,852, 0,850, **0,854**, 0,852, 0,851, 0,852.

Model iki dosya olarak kaydedilir: ağırlıklar `models/topic_model.npz`, sözlük ve ayarlar
`models/topic_model.json`. Yüklemede `pickle` kullanılmaz (`allow_pickle=False`), bu nedenle
model dosyası kod çalıştıramaz.

## 7. Sohbet konusu ve arama sorgusu

Her genel konu için `skor = eski_skor · 0,7 + yeni_olasılık` hesaplanır. Toplam içindeki payı
%15'i geçen en fazla 3 konu sohbetin genel konusunu oluşturur. %15, üç konulu bir sohbette iki
mesaj önce konuşulan konunun (ağırlığı yaklaşık 0,49'a inmiş) hâlâ temada kalabilmesi için
seçildi.

Tema ifadesi konuların rolüne göre kurulur: Kitaplar "ortam", Bilim "niteleyici", diğerleri
"alan" konusudur:

* Kitaplar → "kitaplar"
* Kitaplar + Bilim → "bilimsel kitaplar"
* Kitaplar + Bilim + Biyoloji → "biyoloji hakkında bilimsel kitaplar"
* Tarih + Kitaplar → "tarih kitapları"

Sohbet tek bir konuya odaklanmışsa ve bir alt konu baskınsa tema alt konuya daralır ("kuantum
bilgisayarlar"). Arama sorgusu tema ifadesidir; tema kısaysa son mesajdan modelde yüksek
ağırlığa sahip en fazla 2 sözcük eklenir. Çekimli fiiller eklenmez; geçmiş zaman eki Türkçe
ünsüz uyumuna göre kontrol edilir ("kuantum" fiil sanılmaz, "kaptım" elenir).

## 8. İnternet araması ve veritabanı

* Arama önce Türkçe Wikipedia API'sinde yapılır, sonuç yoksa DuckDuckGo denenir. Zaman aşımı,
  yeniden deneme, `Retry-After` desteği ve yalnızca bu iki alan adına ait bağlantıları kabul
  eden filtre vardır. Bağlantı yoksa 120 saniye boyunca yeniden denenmez; sınıflandırma ve kayıt
  devam eder.
* Aynı sorgu 24 saat boyunca veritabanındaki önbellekten cevaplanır.
* SQLite tabloları: `metinler`, `sohbet_konulari`, `arama_sonuclari`, `sessions`,
  `model_metadata`, `search_cache`. Yabancı anahtarlar, indeksler, parametreli sorgular ve
  `PRAGMA user_version` ile şema sürümü kullanılır.

## 9. Sonuçlar

Test seti yalnızca bir kez, tüm seçimler bittikten sonra ölçüldü.

### 9.1 Genel ölçümler

| Ölçüm | Değer |
|---|---|
| Macro F1 (güven eşiğiyle) | 0,830 |
| Macro F1 (eşiksiz) | 0,850 |
| Doğruluk (eşikli / eşiksiz) | 0,882 / 0,898 |
| Alt konu doğruluğu (genel konu doğruyken) | 0,931 |
| ECE | 0,022 |
| Görülmemiş konuları reddetme oranı | 0,884 |
| Yalnızca kavram adı verildiğinde (1-3 sözcük) doğruluk / macro F1 | 0,621 / 0,544 |
| Harici biyoloji / Osmanlı tarihi (eşiksiz) | 0,688 / 0,420 |
| Tek metin tahmin süresi (medyan / %95) | 0,70 ms / 0,88 ms |

### 9.2 Konu bazında (test, eşikli)

| Konu | Precision | Recall | F1 | Örnek |
|---|---|---|---|---|
| Fizik | 0,898 | 0,811 | 0,852 | 185 |
| Kimya | 0,873 | 0,705 | 0,780 | 78 |
| Biyoloji | 0,912 | 0,827 | 0,868 | 301 |
| Teknoloji | 0,892 | 0,760 | 0,821 | 271 |
| Bilim | 0,765 | 0,722 | 0,743 | 36 |
| Kitaplar | 0,831 | 0,662 | 0,737 | 148 |
| Spor | 0,974 | 0,879 | 0,924 | 340 |
| Tarih | 0,907 | 0,774 | 0,835 | 190 |
| Diğer | 0,866 | 0,953 | 0,907 | 1.930 |

Eşik, recall'u bir miktar düşürüp precision'u artırır: model emin olmadığında konu iddia
etmek yerine "Belirsiz" der.

### 9.3 Ödev senaryoları ve kabul cümleleri

| Durum | Sonuç |
|---|---|
| Senaryo 1: "Kitaplar hakkında konuşalım." → "Bilim ile ilgili neler var?" → "Biyoloji hakkında ne önerirsin?" | Geçti. Temalar: "kitaplar", "bilimsel kitaplar", "biyoloji hakkında bilimsel kitaplar" |
| Senaryo 2: "Kuantum bilgisayarlar nasıl çalışır?" → "Kuantum mekaniği neyi açıklar?" | Geçti. Teknoloji > Kuantum Bilgisayarlar, ardından Fizik > Kuantum Mekaniği |
| "kuantum" (tek sözcük) | Teknoloji %77, ilişkili konu Fizik %23; alt konu Kuantum Bilgisayarlar |
| "Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir." | Fizik > Kuantum Mekaniği (%100) |
| "Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir." | Teknoloji > Kuantum Bilgisayarlar (%99) |
| "Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı." | Kitaplar > Edebiyat, Roman ve Şiir (%99,6) |
| "Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder." | **Belirsiz** (en olası Bilim, %55 < 0,60) |
| "Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir." | Biyoloji > Genetik (%86) |

Bu durumlar `python proje.py --degerlendir` ile yeniden üretilebilir ve testlerde de
denetlenir.

## 10. Bilinen sınırlılıklar

1. **Bilim sınıfı zayıf.** Bu sınıf yalnızca bilim felsefesi ve yöntem kartlarından oluşur
   (182 eğitim örneği). "Bilim insanları hipotez test eder" cümlesinde en olası konu doğru
   (Bilim) olsa da güven eşiğin altında kalır ve sonuç "Belirsiz" olur. Sohbet takibi
   olasılıkları eşikten bağımsız biriktirdiği için bu cümle yine de "bilimsel kitaplar"
   temasına katkı verir.
2. **Farklı üslupta düşük başarı.** Model kısa tanım cümleleriyle eğitildiği için uzun
   ansiklopedik paragraflarda başarı düşer (Osmanlı tarihi %42; yanlışların çoğu "Diğer").
3. **Kısa girdiler.** Yalnızca bir kavram adı verildiğinde doğruluk 0,62'dir. Konu adlarının
   kendisi (biyoloji, fizik, kuantum…) eğitim örneği olarak eklendiği için tanınır.
4. **Coğrafya gibi komşu konular.** Görülmemiş konulardan en çok coğrafya ve balıkçılık
   metinleri yanlışlıkla bir konuya atanır (%21-32).
5. **Alt konu çeşitliliği.** "Kitaplar hakkında konuşalım" gibi genel bir cümlede en olası alt
   konu Çizgi Roman çıkar; alt konular listelendiği ve olasılıkları gösterildiği için kullanıcı
   dağılımı görebilir.
6. **Sorguda fiil kalması.** Geniş zaman fiilleri ("çalışır") filtrelenmez ve sorguya
   eklenebilir.
7. **İnternet.** Web araması Wikipedia/DuckDuckGo erişimine bağlıdır; bu ortamda canlı arama
   testi çalıştırılamadı, arama katmanı sahte HTTP yanıtlarıyla test edildi.

## 11. Testler

`python -m pytest`: 165 test geçer, canlı ağ testi atlanır (`NLP_NETWORK_TESTS=1` ile
çalışır), 1 test bilinen sorun olarak işaretlidir (10.1). Testler kendi yazılan algoritmaları
(TF-IDF formülü, Naive Bayes, softmax, sıcaklık, metrikler, gruplu bölme), veri hazırlamayı,
modelin kaydedilip yüklenmesini, sohbet takibini, sorgu üretimini, web arama hata durumlarını,
veritabanını ve konsol uygulamasının tamamını kapsar. `ruff` ve `mypy` hatasız geçer.

## 12. Geliştirme önerileri

* Bilim ve Kitaplar sınıfları için bu konularda gerçek kullanıcı cümleleri toplamak.
* Uzun paragraflar için cümle bazında tahmin edip olasılıkları birleştirmek.
* Konu bazında ayrı güven eşiği seçmek (az örnekli sınıflarda daha düşük eşik).
* Coğrafya gibi karışan konular için "Diğer" sınıfına daha çeşitli örnekler eklemek.
