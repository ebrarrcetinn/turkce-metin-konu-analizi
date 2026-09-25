# Türkçe Metin Konu Analizi — Proje Raporu

Hazırlayan: Ebrar Cemre Çetin
Tarih: 27.09.2026

## 1. Amaç

Bu projede amacım, kullanıcının yazdığı Türkçe bir metnin genel konusunu ve alt konularını bulan,
birden fazla mesajdan oluşan bir sohbetin genel konusunu takip eden, bu konu için internette
arama yapan ve metinleri, konuları, sorguları ve arama sonuçlarını veritabanına kaydeden bir
konsol uygulaması geliştirdim. Ödevde istendiği gibi sınıflandırma algoritmalarını hazır
makine öğrenmesi kütüphaneleri kullanmadan kendim yazdım; `numpy` ve `scipy`'yi yalnızca
dizi ve seyrek matris yapısı olarak kullandım.

Örneğin kullanıcı sırasıyla "Kitaplar hakkında konuşalım.", "Bilim ile ilgili neler var?" ve
"Biyoloji hakkında ne önerirsin?" yazdığında program her mesajın konusunu (Kitaplar, Bilim,
Biyoloji) bulur, sohbetin konusunu adım adım "kitaplar" → "bilimsel kitaplar" → "biyoloji
hakkında bilimsel kitaplar" olarak günceller ve son ifadeyle internette arama yapar. Ben bu
akışın her adımını ayrı sınıflar halinde kurdum ve her adımı ayrı ayrı test ettim.

## 2. Sistem tasarımı

Bu bölümde amacım programı hangi parçalara ayırdığımı ve bu parçaların birbirine nasıl
bağlandığını göstermek.

Programı nesne yönelimli olarak katmanlara ayırdım. Her katman yalnızca kendi işini yapıyor;
örneğin konsol sınıfı hiçbir hesap yapmıyor, yalnızca girdi alıp sonucu yazıyor. Böylece her
parçayı ayrı ayrı test edebildim.

| Katman | Sınıflar | Görev |
|---|---|---|
| Konsol | `ConsoleApp`, `ModelPreparer` (`proje.py`) | Girdiyi alıp sonucu yazıyorum; model yoksa veriyi indirip eğitiyorum |
| Uygulama servisi | `ChatService` | Bir mesaj için tüm adımları sırayla çalıştırıyor, hataları yalıtıyorum |
| Model | `TopicModel`, `Trainer`, `Evaluator` | Tahmin yapıyor, modeli eğitiyor ve ölçüyorum |
| Algoritmalar | `TfidfVectorizer`, `MultinomialNaiveBayes`, `SoftmaxRegression`, `ClassificationReport` | Kendi yazdığım ML bileşenleri |
| Veri | `DatasetBuilder`, `DataSource` | Veriyi indirip doğruluyor, temizliyor ve bölüyorum |
| Sohbet | `ConversationTracker`, `Theme` | Sohbet konusunu ve tema ifadesini buluyorum |
| Arama | `WebSearchClient` | Wikipedia'da, gerekirse DuckDuckGo'da arıyorum |
| Kayıt | `Database` | SQLite tablolarını, şema sürümünü ve önbelleği yönetiyorum |

Programı tek komutla çalışacak şekilde tasarladım: `python proje.py`. Bilgisayarda eğitilmiş
model yoksa önce veriyi indiriyor, veri setini hazırlıyor ve modeli eğitiyorum (yaklaşık
1 dakika), sonra sohbet döngüsüne geçiyorum.

## 3. Bir mesajın baştan sona işlenişi (örnek)

Bu bölümde amacım sistemin nasıl çalıştığını tek bir örnek cümle üzerinden adım adım
göstermek. Aşağıdaki sayıların hepsi eğittiğim modelin gerçek çıktıları.

> Kullanıcı: "Kübitler, süperpozisyon sayesinde aynı anda birden fazla durumu temsil edebilir!"

**Adım 1 — Ön işleme.** Metni küçük harfe çeviriyor, noktalama işaretlerini atıyorum:

    kübitler süperpozisyon sayesinde aynı anda birden fazla durumu temsil edebilir

Sözcük özellikleri için her sözcüğün ilk 5 harfini alıyorum (F5 kökleme):

    kübit süper sayes aynı anda birde fazla durum temsi edebi

**Adım 2 — Sayıya çevirme (TF-IDF).** Model sayılarla çalıştığı için metni bir vektöre
çeviriyorum. Bu cümleden 15 sözcük özelliği ("kübit", "fazla durum" gibi) ve 241 karakter
parçası ("kübi", "üper" gibi) çıkıyor. Toplam 224.759 olası özellikten yalnızca bunlar sıfırdan
farklı. Her özelliğin ağırlığını ne kadar ayırt edici olduğuna göre belirliyorum: "kübit" az
metinde geçtiği için IDF değeri 7,66, "durum" çok yerde geçtiği için 5,10.

**Adım 3 — Model skoru.** Eğittiğim softmax regresyon modelinde 39 sınıfın ("Fizik > Optik",
"Teknoloji > Kuantum Bilgisayarlar", …, "Diğer") her biri için her özelliğin bir ağırlığı
var. Örneğin eğitim sonunda "kübit" sözcüğünün ağırlığı "Teknoloji > Kuantum Bilgisayarlar"
için +1,64, "Fizik > Kuantum Mekaniği" için −0,75 oldu. Metindeki özelliklerin ağırlıklarını
toplayarak her sınıfa bir skor hesaplıyorum:

| Sınıf | Skor |
|---|---|
| Teknoloji > Kuantum Bilgisayarlar | +1,02 |
| Teknoloji > Nanoteknoloji | −2,05 |
| Diğer | −2,74 |

**Adım 4 — Olasılık.** Skorları softmax fonksiyonuyla toplamı %100 olan olasılıklara
çeviriyorum: Kuantum Bilgisayarlar %96,4, Nanoteknoloji %1,3, Diğer %0,5.

**Adım 5 — Genel konu ve alt konu.** Genel konunun olasılığı, alt konularının toplamı:
Teknoloji = %96,4 + %1,3 + … = **%97,8**. Bu değer güven eşiğinin (%60) üstünde olduğu için
sonucu kesin kabul ediyorum. Alt konuyu Teknoloji içindeki payla veriyorum: Kuantum
Bilgisayarlar %98,5.

**Adım 6 — Sohbet konusu ve arama.** Bu mesajın olasılıklarını sohbet skorlarına ekliyorum
(bkz. bölüm 8). Sohbetin ilk mesajı olduğu ve tek bir alt konu baskın olduğu için tema
"Teknoloji > Kuantum Bilgisayarlar" oluyor. Arama sorgusunu tema adına mesajdaki en ayırt
edici iki sözcüğü ekleyerek kuruyorum: **"kuantum bilgisayarlar kübitler sayesin"**.
"kübitler" modelde bu konu için en yüksek ağırlığa sahip sözcük. "sayesinde" sözcüğünün
sonundaki "-de" bulunma ekini sorgu için kırpıyorum.

**Adım 7 — Kayıt.** Metni, sonucu, sohbet konusunu, sorguyu ve internetten gelen sonuçları
SQLite veritabanına yazıyorum.

## 4. Metin ön işleme

Bu bölümde amacım metni modele vermeden önce neden ve nasıl sadeleştirdiğimi anlatmak.
Aynı ön işlemeyi hem eğitimde hem tahminde kullanıyorum; aksi halde model eğitimde
gördüğünden farklı biçimde metin alır ve başarısı düşer.

* **Türkçe küçük harf.** Python'un `str.lower()` fonksiyonu "IŞIK" sözcüğünü "işik" yapıyor;
  doğrusu "ışık". Bu yüzden I→ı ve İ→i dönüşümünü önce elle yapan `turkish_lower` fonksiyonunu
  yazdım.
* **Normalizasyon.** Bozuk karakter kodlamasını onarıyor, Unicode biçimini tekleştiriyor,
  HTML etiketlerini, bağlantıları, @kullanıcı adlarını, emojileri ve rakamları atıyorum.
  Kesme işaretinden sonraki ekleri de atıyorum: "Ankara'da" → "ankara".
* **F5 kökleme.** Türkçede ekler sözcüğün sonuna geldiği için ilk 5 harf çoğu zaman kökü
  koruyor: "hücreler", "hücrenin" → "hücre"; "kütüphaneden" → "kütüp". Sözlük gerektirmeyen
  basit bir yöntem olduğu ve Türkçe metinlerde iyi sonuç verdiği için bunu seçtim.
* **Stopword'ler.** "ve", "bir", "için" gibi her konuda geçen sözcükleri özelliklerden
  çıkarıyorum.
* **Boş girdiler.** "!!!", "12345" gibi anlamlı sözcük içermeyen girdileri sınıflandırmıyor,
  kullanıcıya uyarı veriyorum.

## 5. Veri

Bu bölümde amacım hangi veriyi kullandığımı, bu veriyi nasıl hazırladığımı ve testte
kopya çekmeyi (veri sızıntısını) nasıl önlediğimi göstermek.

Eğitim için GitHub'daki **Türkçe Tabu Veri Seti**'ni (MIT lisanslı, 150 kategori, 37.278 kart)
kullandım. Her kart bir kavram ve kısa açıklamasından oluşuyor. Örnek bir kart ("genetik"
kategorisi):

> **alel** — bir genin kromozomdaki lokusunda bulunan alternatif formlarından biri.

Eğitim metni olarak "kavram + açıklama" birleşimini aldım ve kartın kategorisini ödevdeki
konulara eşledim: "genetik" → Biyoloji > Genetik, "futbol" → Spor > Futbol, "osmanli" →
Tarih > Osmanlı Tarihi gibi. Veriyi her seferinde aynı sürüm gelsin diye sabit bir commit
adresinden indiriyor ve her dosyayı SHA-256 özetiyle doğruluyorum.

| Adım | Sayı |
|---|---|
| Ham kart | 37.278 |
| Birden fazla konuya girebilen 31 kategori nedeniyle çıkarılan | 7.693 |
| Temizlik sonrası | 29.585 |
| Eğitim / doğrulama / test | 16.023 / 3.458 / 3.479 |
| Görülmemiş konu — eşik seçimi / son ölçüm | 2.785 / 3.275 |
| Harici değerlendirme (biyoloji + Osmanlı tarihi cümleleri) | 800 |

Veriyle ilgili aldığım kararlar:

* **Belirsiz kategorileri çıkardım.** Örneğin "anatomi" biyolojiye, "jeoloji" bilime çok
  yakın. Bunları "Diğer" olarak eğitseydim model biyoloji metinlerine "Diğer" demeyi
  öğrenirdi; bu yüzden 31 kategoriyi tamamen dışarıda bıraktım.
* **Kuantumu ikiye ayırdım.** Veri setinde yalnızca bir "kuantum" kategorisi var. Ödevdeki
  Fizik / Teknoloji ayrımını öğretebilmek için terimi kuantum hesaplamaya ait olan kartları
  (kübit, kuantum kapısı, transmon…) Teknoloji > Kuantum Bilgisayarlar olarak, kalanları
  Fizik > Kuantum Mekaniği olarak etiketledim. Örnek:
  - "süperpozisyon — kuantum sisteminin ölçülene kadar olası tüm durumlarda eşzamanlı
    bulunabilmesi ilkesi" → Fizik > Kuantum Mekaniği
  - "topolojik kuantum bilgisayarı — hata toleransını … anyonların düğümlenmesiyle sağlayan
    mimarisi" → Teknoloji > Kuantum Bilgisayarlar
* **Terim gruplu bölme yaptım.** Aynı kavram birden fazla kartta geçebildiği için aynı
  kavramın kartlarını hep aynı bölmeye koydum. Aksi halde model testte ezberlediği kavramı
  görür ve başarı olduğundan yüksek çıkardı. Kontrol ettiğimde eğitim ile test arasında ortak
  kavram sayısı 0, neredeyse aynı metin oranı %0,06 çıktı.
* **Modelin hiç görmediği konular ayırdım.** "Diğer" kategorilerinin %35'ini (ör. balıkçılık,
  kaligrafi) eğitimden tamamen çıkardım. Böylece modelin tanımadığı bir konuda "emin değilim"
  deyip diyemediğini ölçebildim.
* **Konu adlarını ekledim.** Tabu oyununda kartın kendi kategori adı yasaklı sözcük olduğu
  için "biyoloji" sözcüğü biyoloji kartlarında neredeyse hiç geçmiyor. İlk denememde model tek
  başına "biyoloji" yazılınca Teknoloji > Nanoteknoloji dedi, çünkü "-oloji" harf parçası
  nanoteknoloji kartlarında sık geçiyordu. Her alt konu için "Genetik", "Biyoloji", "Biyoloji
  Genetik" gibi 114 kısa örneği eğitim verisine ekledim; şimdi "biyoloji" girdisi %98,8 ile
  Biyoloji çıkıyor. Bu örnekler yalnızca eğitim verisinde, test sonuçlarını etkilemiyor.

## 6. Kullandığım algoritmalar

Bu bölümde amacım kendi yazdığım algoritmaları, formülleriyle ve örneklerle açıklamak.

### 6.1 TF-IDF

İki vektörü yan yana birleştirdim: F5 köklerinden sözcük 1-2 gram ("kübit", "fazla durum") ve
harf 2-5 gram ("kü", "kübi"). Harf parçaları Türkçedeki ekleri yakalıyor: "kitap", "kitaplar"
ve "kitabın" ortak parçalar üzerinden birbirine bağlanıyor.

* IDF = ln((1 + n) / (1 + df)) + 1. df, bir terimin kaç metinde geçtiği. Az metinde geçen
  terimi daha ayırt edici sayıyorum (yukarıdaki örnekte "kübit" 7,66, "durum" 5,10).
* TF = 1 + ln(tf). Bu sayede bir sözcüğün 10 kez geçmesi 10 kat değil yaklaşık 3,3 kat etki
  ediyor.
* Her vektörü L2 normuyla 1 uzunluğa getiriyorum; böylece uzun ve kısa metinler aynı ölçekte
  kalıyor.

Sözlüğü ve IDF değerlerini yalnızca eğitim verisinden öğreniyorum.

### 6.2 Naive Bayes (karşılaştırma modeli)

Bayes kuralıyla her sınıf için log P(sınıf) + Σ değer · log P(özellik | sınıf) skorunu
hesaplıyorum. Eğitimde görülmemiş bir özelliğin olasılığı sıfır olmasın diye Laplace
düzeltmesi (α) ekliyorum. Klasik değer α = 1, ama 200 bine yakın özellikte bu değer bütün
sınıfları birbirine benzetti. Doğrulama setinde denediğim değerler:

| α | Doğrulama macro F1 |
|---|---|
| 1 | 0,079 |
| 0,1 | 0,515 |
| 0,01 | 0,822 |
| **0,001** | **0,833** |

### 6.3 Softmax regresyon (ana model)

Her sınıf için bir ağırlık sütunu tutuyorum. Skor = x · W + b; olasılık = softmax(skor).
Eğitimde yaptığım şey, doğru sınıfın olasılığını artıracak şekilde W ve b'yi küçük adımlarla
düzeltmek.

1. Ağırlıkları sıfırdan başlatıyorum; başta her sınıf eşit olasılıklı.
2. Veriyi 256 örneklik gruplar halinde veriyorum. Her grupta modelin tahmin ettiği
   olasılıklarla doğru cevap arasındaki farkı (p − y) hesaplıyorum. Örneğin doğru sınıfa
   %30 olasılık verildiyse fark −0,70, yanlış bir sınıfa %40 verildiyse +0,40.
3. Bu farkla ağırlıkların türevini (xᵀ · (p − y) + λW) buluyor ve ağırlıkları ters yönde
   biraz güncelliyorum. Güncellemeyi Adam yöntemiyle yapıyorum; Adam her ağırlık için adım
   büyüklüğünü kendisi ayarlıyor, nadir sözcüklerde daha büyük adım atıyor.
4. **Sınıf ağırlıkları.** "Diğer" sınıfında 8.831, Bilim'de yalnızca 182 örnek var. Model küçük
   sınıfları görmezden gelmesin diye her sınıfın hatasını n / (K · n_k) ile ağırlıklandırdım;
   böylece Bilim örneklerindeki hatayı yaklaşık 48 kat ağır sayıyorum.
5. **L2 düzenlileştirme.** Ağırlıkların aşırı büyümesini (ezberlemeyi) küçük bir cezayla
   engelliyorum.
6. **Erken durdurma.** Verinin tamamı bir kez dolaşılınca bir tur (epoch) tamamlanıyor. Her
   turdan sonra modeli doğrulama setinde ölçüyorum. 3 tur üst üste iyileşme olmazsa eğitimi
   durdurup en iyi turdaki ağırlıkları alıyorum.
7. Seyrek veride hız için yalnızca o grupta geçen özelliklerin ağırlıklarını güncelliyorum;
   200 bin × 39'luk matrisin tamamını her adımda dolaşmak çok daha yavaş olurdu.

### 6.4 Kalibrasyon

Burada amacım, modelin "%80 eminim" dediği tahminlerin gerçekten yaklaşık %80'inin doğru
olmasını sağlamak. Aksi halde "Belirsiz" kararı için koyduğum eşik anlamsız kalır. Bunun için temperature scaling
kullandım: skorları bir T sayısına bölüyorum. T'yi doğrulama setinde en iyi sonucu verecek
şekilde altın oran aramasıyla buldum: T = 0,72. T'nin 1'den küçük çıkması, sınıf ağırlıkları ve
düzenlileştirme yüzünden modelin olduğundan çekingen olduğunu ve olasılıkların biraz
keskinleştirilmesi gerektiğini gösteriyor. Sonuçta kalibrasyon hatasını (ECE) testte 0,022
olarak ölçtüm.

### 6.5 Güven eşiği

Burada amacım modelin ne zaman "emin değilim" diyeceğine karar vermek. 0,20 ile 0,90
arasındaki eşikleri denedim. Eşik düşük olursa model tanımadığı konulara da konu atıyor: 0,20 eşiğinde "Merhaba" yazınca %37 ile Kitaplar diyordu. Eşik yüksek olursa doğru
konuları da "Belirsiz" sayıyor. Eşiği, doğrulama seti ile modelin hiç görmediği konuları
birlikte kullanarak seçtim:

| Eşik | 0,20 | 0,40 | 0,50 | **0,60** | 0,70 | 0,80 | 0,90 |
|---|---|---|---|---|---|---|---|
| Macro F1 | 0,752 | 0,766 | 0,774 | **0,782** | 0,778 | 0,770 | 0,739 |

0,60'ı seçtikten sonra "Merhaba" ve "bugün hava çok güzel" gibi girdilere artık konu
atanmıyor.

### 6.6 Metrikler

Doğruluk, her sınıf için precision / recall / F1, macro F1, karmaşıklık matrisi ve kalibrasyon
hatasını (ECE) kendim hesapladım. "Diğer" sınıfı çok büyük olduğu için başarıyı doğruluk
yerine macro F1 ile ölçtüm. Macro F1 her konuya eşit ağırlık veriyor; yalnızca "Diğer" demeyi
öğrenen bir model yüksek doğruluk alabilir ama macro F1'i düşük çıkar.

## 7. Eğitim süreci ve model seçimi

Bu bölümde amacım modeli nasıl eğittiğimi ve iki algoritma arasından neden softmax
regresyonu seçtiğimi göstermek. Naive Bayes'i ve softmax regresyonu aynı veriyle eğitip
doğrulama setinde karşılaştırdım.

| Aşama | Sonuç |
|---|---|
| Naive Bayes, doğrulama macro F1 | 0,833 |
| Softmax regresyon, doğrulama macro F1 | **0,854** (en iyi tur: 9) |
| Seçilen model | Softmax regresyon |
| Sıcaklık (T) | 0,717 |
| Güven eşiği | 0,60 |
| Son eğitim | Eğitim + doğrulama verisi birlikte, 9 tur |
| Toplam süre (2 çekirdekli CPU) | yaklaşık 55 saniye |

Softmax regresyonun turlara göre doğrulama macro F1 değerleri:

| Tur | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0,710 | 0,824 | 0,842 | 0,848 | 0,848 | 0,850 | 0,852 | 0,850 | **0,854** | 0,852 | 0,851 | 0,852 |

9. turdan sonra 3 tur boyunca iyileşme olmadığı için eğitimi 12. turda durdurdum ve 9. turun
ağırlıklarını aldım. Seçimleri
bitirdikten sonra modeli eğitim ve doğrulama verisinin toplamıyla aynı ayarlarla 9 tur yeniden
eğittim, çünkü daha fazla örnek daha iyi model demek.

Modeli iki dosya olarak kaydediyorum: ağırlıklar `models/topic_model.npz`, sözlük ve ayarlar
`models/topic_model.json`. `pickle` kullanmadım, çünkü pickle dosyası açılırken kod
çalıştırabilir. Aynı veri ve aynı rastgelelik tohumuyla eğitim her seferinde birebir aynı
modeli üretiyor; bunu ağırlıkların SHA-256 özetini karşılaştırarak doğruladım.

## 8. Sohbet konusu ve arama sorgusu

Bu bölümde amacım sohbetin genel konusunu nasıl takip ettiğimi ve arama sorgusunu nasıl
ürettiğimi göstermek.

Her mesajdan sonra her genel konu için skoru şöyle güncelliyorum:

    yeni skor = eski skor × 0,7 + bu mesajdaki olasılık

Böylece son mesaj en etkili oluyor; bir önceki 0,7, ondan önceki 0,49 ağırlıkla katkı
veriyor. Toplam içindeki payı %15'i geçen en fazla 3 konuyu sohbetin genel konusu olarak
alıyorum. Ödevdeki 1. senaryoda
gerçek değerler şöyle:

| Mesaj | Mesajın konusu | Sohbet skorları | Paylar | Tema |
|---|---|---|---|---|
| Kitaplar hakkında konuşalım. | Kitaplar (%79) | Kitaplar 0,79 | Kitaplar %87 | kitaplar |
| Bilim ile ilgili neler var? | Bilim (%91) | Bilim 0,92, Kitaplar 0,56 | Bilim %57, Kitaplar %35 | bilimsel kitaplar |
| Biyoloji hakkında ne önerirsin? | Biyoloji (%60) | Bilim 0,67, Biyoloji 0,61, Kitaplar 0,41 | Bilim %32, Biyoloji %29, Kitaplar %19 | biyoloji hakkında bilimsel kitaplar |

Tabloda Kitaplar skorunun 0,79 → 0,56 → 0,41 şeklinde yavaşça azaldığını görüyorum. İlk
denememde eşik %20 idi; üçüncü mesajda Kitaplar'ın payı %19'a düştüğü için temadan çıkıyordu.
Eşiği %15'e indirdim, çünkü 0,7 azalma katsayısıyla iki mesaj önce konuşulan bir konunun payı
üç konulu bir sohbette tam bu aralığa düşüyor.

Tema cümlesini konuların rolüne göre kuruyorum. Kitaplar'ı içeriğin türü (cümlenin başı),
Bilim'i niteleyici (sıfat), diğerlerini alan konusu olarak tanımladım. Bu kuralları ödevdeki
örneklere özel yazmadım, her konu birleşimi için çalışıyorlar:

* Kitaplar → "kitaplar"
* Kitaplar + Bilim → "bilimsel kitaplar"
* Kitaplar + Bilim + Biyoloji → "biyoloji hakkında bilimsel kitaplar"
* Kitaplar + Tarih → "tarih kitapları"
* Bilim + Biyoloji → "biyoloji alanında bilimsel araştırmalar"

Sohbet tek bir konudaysa ve bir alt konu baskınsa temayı alt konuya daraltıyorum: "Kuantum
bilgisayarlar nasıl çalışır?" → tema "kuantum bilgisayarlar". Arama sorgusunu tema
ifadesinden oluşturuyorum. Tema kısaysa son mesajdan modelde yüksek ağırlığa sahip en fazla 2 sözcük ekliyorum.
Çekimli fiilleri eklemiyorum. Başta programım "kuantum" sözcüğünü "-tum" ile bittiği için
geçmiş zaman fiili sanıyordu ("tuttum" gibi). Türkçe ünsüz uyumuna göre "-tı" eki yalnızca sert
ünsüzden sonra gelebildiği için kuralı buna göre düzelttim.

## 9. İnternet araması ve veritabanı

Bu bölümde amacım internette nasıl arama yaptığımı ve her şeyi veritabanına nasıl
kaydettiğimi anlatmak.

* Aramayı önce Türkçe Wikipedia API'sinde yapıyorum; sonuç çıkmazsa DuckDuckGo'yu deniyorum.
  İkisi de API anahtarı istemiyor, bu yüzden projede gizli bilgi yok.
* Sunucu cevap vermezse 6 saniye sonra vazgeçiyorum. Geçici hatalarda (429, 5xx) biraz
  bekleyip tekrar deniyorum. Yalnızca wikipedia.org ve duckduckgo.com adreslerinden gelen
  bağlantıları kabul ediyorum.
* İnternet yoksa sınıflandırmaya ve kayda devam ediyor, kullanıcıya "İnternete erişilemedi"
  yazıyorum. Her mesajda zaman aşımı beklenmesin diye 2 dakika boyunca tekrar denemiyorum.
* Aynı sorgu 24 saat içinde tekrar sorulursa sonuçları veritabanındaki önbellekten
  getiriyorum.

Veritabanı tabloları ve 1. senaryodaki bir mesajın kayıtları:

| Tablo | Ne tutuyor | Örnek |
|---|---|---|
| `metinler` | Mesaj, genel konu, güven, alt konular, tüm olasılıklar | "Bilim ile ilgili neler var?", Bilim, 0,915 |
| `sohbet_konulari` | Mesajdan sonraki sohbet konusu ve sorgu | "Bilim + Kitaplar", "bilimsel kitaplar" |
| `arama_sonuclari` | İnternetten gelen başlık, özet, bağlantı ve sıra | Wikipedia sonuçları |
| `sessions` | Her sohbet oturumu | `sıfırla` komutu yeni oturum açar |
| `model_metadata` | Kayıtları üreten modelin sürümü ve metrikleri | v2.0.0, softmax_regression |
| `search_cache` | Önbellek | 24 saatlik sonuçlar |

Tabloları yabancı anahtarlarla birbirine bağladım; örneğin her arama sonucu kendi sohbet
konusu kaydına, o da kendi mesajına bağlı. Tüm sorguları parametreli yazdım, böylece kullanıcının
yazdığı metin hiçbir zaman doğrudan SQL'e eklenmiyor. Şema değişikliklerini sürüm numarasıyla
yönetiyorum: eski bir veritabanı dosyası silinmeden otomatik güncelleniyor.

## 10. Sonuçlar

Bu bölümde amacım modelin ne kadar başarılı olduğunu, eğitimde hiç görmediği verilerle
ölçerek göstermek.

Test setini yalnızca bir kez, bütün seçimleri bitirdikten sonra ölçtüm.

### 10.1 Genel ölçümler

| Ölçüm | Değer |
|---|---|
| Macro F1 (güven eşiğiyle) | 0,830 |
| Macro F1 (eşiksiz) | 0,850 |
| Doğruluk (eşikli / eşiksiz) | 0,882 / 0,898 |
| Alt konu doğruluğu (genel konu doğruyken) | 0,931 |
| Kalibrasyon hatası (ECE) | 0,022 |
| Hiç görülmemiş konuları reddetme oranı | 0,884 |
| Yalnızca kavram adı verildiğinde (1-3 sözcük) doğruluk / macro F1 | 0,621 / 0,544 |
| Harici biyoloji / Osmanlı tarihi cümleleri (eşiksiz) | 0,688 / 0,420 |
| Tek metin tahmin süresi (medyan / %95) | 0,70 ms / 0,88 ms |

### 10.2 Konu bazında (test, eşikli)

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

Precision'ın recall'dan yüksek olması koyduğum eşiğin etkisi: model emin olmadığında yanlış
konu söylemek yerine "Belirsiz" diyor.

### 10.3 Ödev senaryoları ve kabul cümleleri

| Girdi | Sonuç |
|---|---|
| Senaryo 1: "Kitaplar hakkında konuşalım." → "Bilim ile ilgili neler var?" → "Biyoloji hakkında ne önerirsin?" | Geçti. Temalar: "kitaplar", "bilimsel kitaplar", "biyoloji hakkında bilimsel kitaplar" |
| Senaryo 2: "Kuantum bilgisayarlar nasıl çalışır?" → "Kuantum mekaniği neyi açıklar?" | Geçti. Teknoloji > Kuantum Bilgisayarlar, ardından Fizik > Kuantum Mekaniği |
| "kuantum" (tek sözcük) | Teknoloji %77, ilişkili konu Fizik %23; alt konu Kuantum Bilgisayarlar |
| "Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir." | Fizik > Kuantum Mekaniği (%100) |
| "Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir." | Teknoloji > Kuantum Bilgisayarlar (%99) |
| "Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı." | Kitaplar > Edebiyat, Roman ve Şiir (%99,6) |
| "Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder." | **Belirsiz** (en olası Bilim, %55 < %60) |
| "Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir." | Biyoloji > Genetik (%86) |
| "bugün hava çok güzel" | Taksonomi dışı (Diğer %80) |

Bu sonuçları `python proje.py --degerlendir` komutuyla yeniden üretebiliyorum; testlerde de
kontrol ediyorum.

## 11. Karşılaştığım sorunlar ve çözümlerim

Bu bölümde amacım geliştirme sırasında karşılaştığım sorunları, nedenlerini ve nasıl
çözdüğümü göstermek.

| Sorun | Neden | Çözüm |
|---|---|---|
| Naive Bayes macro F1 0,08 çıktı | α = 1, 200 bin özellikte gerçek sayımları bastırıyordu | α'yı doğrulama setinde denedim, 0,001 seçtim (0,833) |
| "biyoloji" girdisi Teknoloji çıktı | Kategori adı Tabu kartlarında yasaklı sözcük; veride yok | Konu ve alt konu adlarını eğitim örneği olarak ekledim |
| 3. mesajda "kitaplar" temadan düştü | Kitaplar'ın payı %19'a iniyordu, eşik %20 idi | Eşiği azalma katsayısına göre hesaplayıp %15 yaptım |
| "Merhaba" Kitaplar olarak sınıflandı | Güven eşiği yalnızca bilinen konularla seçilince 0,20'ye düşüyordu | Eşiği modelin hiç görmediği konuları da katarak seçtim (0,60) |
| "kuantum" sorgudan atılıyordu | "-tum" sonu geçmiş zaman eki sanılıyordu | Kuralı Türkçe ünsüz uyumuna göre düzelttim |
| Test başarısı gerçekte olduğundan yüksek çıkabilirdi | Aynı kavramın kartları hem eğitimde hem testte olabilirdi | Terim gruplu bölme yaptım, sızıntıyı ölçüp 0 olduğunu gösterdim |

## 12. Bilinen sınırlılıklar

Bu bölümde amacım sistemin henüz iyi çalışmadığı durumları açıkça belirtmek.

1. **Bilim sınıfı zayıf.** Bu sınıfı yalnızca bilim felsefesi ve yöntem kartlarından
   oluşturabildim (182 eğitim örneği). "Bilim insanları hipotez test eder" cümlesinde en olası konu doğru
   (Bilim %55, ikinci Diğer %37) ama güven eşiğin altında kaldığı için sonuç "Belirsiz".
   Sohbet takibinde olasılıkları eşikten bağımsız biriktirdiğim için bu cümle yine de
   "bilimsel kitaplar" temasına katkı veriyor.
2. **Farklı üslupta düşük başarı.** Modeli kısa tanım cümleleriyle eğittiğim için uzun
   ansiklopedik paragraflarda başarı düşüyor (Osmanlı tarihi %42; yanlışların çoğu "Diğer").
3. **Kısa girdiler.** Yalnızca bir kavram adı verildiğinde doğruluk 0,62. Konu adlarının
   kendisini (biyoloji, fizik, kuantum…) eğitim örneği olarak eklediğim için bunlar tanınıyor.
4. **Coğrafya gibi komşu konular.** Görülmemiş konulardan en çok coğrafya ve balıkçılık
   metinlerini model yanlışlıkla bir konuya atıyor (%21-32).
5. **Genel cümlelerde alt konu.** "Kitaplar hakkında konuşalım" gibi genel bir cümlede en olası
   alt konu Çizgi Roman çıkıyor. Birden fazla alt konuyu olasılıklarıyla birlikte listelediğim
   için kullanıcı dağılımı görebiliyor.
6. **Sorguda fiil kalması.** Geniş zaman fiillerini ("çalışır") henüz filtrelemiyorum; bu
   yüzden sorguya eklenebiliyorlar.
7. **İnternet.** Web araması Wikipedia/DuckDuckGo erişimine bağlı. Arama katmanını sahte HTTP
   yanıtlarıyla test ettim; canlı testi `NLP_NETWORK_TESTS=1` ile çalıştırabiliyorum.

## 13. Testler

Bu bölümde amacım yazdığım kodun doğru çalıştığını nasıl kontrol ettiğimi göstermek.

`python -m pytest` ile 165 test geçiyor; canlı internet testi atlanıyor, 1 testi bilinen sorun
olarak işaretledim (12.1). Testlerim kendi yazdığım algoritmaları (TF-IDF formülü, Naive Bayes,
softmax, sıcaklık, metrikler, gruplu bölme), veri hazırlamayı, modelin kaydedilip yüklenmesini,
sohbet takibini, sorgu üretimini, web aramasının hata durumlarını, veritabanını ve konsol
uygulamasının tamamını kapsıyor. Kodu ayrıca `ruff` ve `mypy` ile kontrol ettim; hata yok.

## 14. Geliştirme önerileri

Bu bölümde amacım projeyi devam ettirsem neleri geliştireceğimi yazmak.

* Bilim ve Kitaplar sınıfları için gerçek kullanıcı cümleleri toplardım.
* Uzun paragraflarda her cümleyi ayrı sınıflandırıp olasılıkları birleştirirdim.
* Her konu için ayrı güven eşiği seçerdim (az örnekli sınıflarda daha düşük eşik).
* Coğrafya gibi karışan konular için "Diğer" sınıfına daha çeşitli örnekler eklerdim.
