# Hata Analizi

Kaynak: `python evaluate.py` → `reports/predictions_test.jsonl`, `reports/evaluation.json`;
`python -m scripts.error_analysis` → `reports/error_analysis.json`. Tüm örnekler gerçek test /
harici set çıktılarıdır (değiştirilmedi; metinlerdeki eksik sözcükler kaynağa aittir).

## 1. Genel tablo (test, n=3.263, eşik 0,75)

* Hata: **345** (%10,6).
  * 134 → "Belirsiz" (güven < 0,75; sistem cevap vermekten kaçındı)
  * 75 → "Diğer" (taksonomi içi metin, taksonomi dışı sayıldı)
  * **136 → güvenle yanlış konu** (gerçek "yanlış cevap" deneyimi)
* 147 hatada doğru sınıf 2. sırada (hataların %43'ü "yakın ıska").

## 2. En çok karıştırılan sınıflar

| Gerçek → Tahmin | Adet | Gerçek örnekler (güven, durum) |
|---|---:|---|
| Teknoloji → Diğer | 37 | "hızlı üs alma farklı zamanlı aktiviteler arasından çakışmayan en çok etkinliği seçme." (0,46, Diğer) — *terim–tanım uyuşmazlığı* ; "finfet akım sızıntısını önlemek ve performansı artırmak için tasarlandığı mimari." (0,81, Diğer) |
| Tarih → Diğer | 35 | "stratigrafi farklı dönemlere ait toprak ve kültür katmanlarını inceleme yöntemi." (0,48, Belirsiz; en olası Tarih) ; "krater ingiltere'de bulunan dev taş blokların dairesel dizildiği gizemli anıt." (0,68, Belirsiz) |
| Biyoloji → Diğer | 33 | "gutasyon yaprak kenarlarından damlalar halinde fazla suyun dışarı atılması" (0,73, Belirsiz; en olası Biyoloji) |
| Kitaplar → Diğer | 31 | "tony stark kolları ve bacakları hareket edebilen eklemli oyuncak figür" (0,33, Belirsiz) |
| Spor → Diğer | 30 | "ara derece tekrarlayan darbeler sonucu kemikte oluşan mikroskobik stres kırığı çatlağı." (0,46, Belirsiz) |
| Fizik → Diğer | 25 | "ses gücü müzik aletlerinin perdesini doğru nota frekansına ayarlama aleti." (0,997, Diğer) — *akustik ↔ müzik aletleri örtüşmesi* |
| Diğer → Teknoloji | 22 | "toptan satış robotik yazılımların piyasada gerçekleştirdiği otomatik hisse işlemleri" (0,81) |
| Diğer → Biyoloji | 20 | "pug cinsi yanaklarındaki turuncu benekleri ve tepeliğiyle bilinen papağan." (1,00) — *etiket gürültüsü* |

Baskın hata biçimi **taksonomi içi ↔ "Diğer"** sınırıdır (satır normalize confusion matrix:
`reports/figures/confusion_matrix_test.png`). Taksonomi içi konular arasında karışma azdır
(ör. Fizik→Kimya 2, Kimya→Fizik 5, Teknoloji→Fizik 4).

## 3. Güvenle yanlış 30 örneğin elle incelemesi

`reports/predictions_test.jsonl` içindeki 136 güvenli hatadan rastgele 30'u (`random_state=0`)
elle incelenip sınıflandırıldı (tek değerlendirici):

| Tür | Adet | Örnek |
|---|---:|---|
| **Meşru alan örtüşmesi** — metin gerçekten tahmin edilen konuyu anlatıyor, kaynak kategorisi farklı | 21 (%70) | "albedo yüzeyin veya gezegenin üzerine düşen güneş ışığını yansıtma oranı." ekoloji→Biyoloji, tahmin Fizik; "ultrason muayenesi ses dalgalarıyla…" evcilhayvanlar→Diğer, tahmin Fizik; "gövdeleme kelimelerdeki tüm çekim ve yapım eklerini atarak kelimeyi algoritması." linguistik→Diğer, tahmin Teknoloji; "yoğuşma gaz fazındaki maddenin…sıvı faza dönüşmesi" kimya, tahmin Fizik |
| **Etiket gürültüsü** — terim ile tanım uyuşmuyor | 6 (%20) | "kuş kumu hayvanın göz çevresindeki çapak ve lekeleri temizleyen steril solüsyon."; "büyü iksiri hızlı koşucuların zamanda yolculuk yapabildiği kozmik enerji alanı" (çizgi roman → Spor) |
| **Gerçek model hatası** | 3 (%10) | "cabasa metal silindirin etrafına sarılmış bilyeli zincirlerin elle perküsyon aletidir." → Kimya (0,83); "yetki devretmek bazı görev ve karar haklarını…" → Teknoloji (0,85) |

Yorum: Ölçülen test hatasının önemli kısmı veri etiketlemesinin (kategori = konu varsayımı)
sınırından kaynaklanıyor; bu, raporlanan metriklerin **gerçek kullanıcı deneyimini kötümser**
yansıtıyor olabileceğini gösterir. Bu tek değerlendiricili, 30 örneklik bir tahmindir; metrikler
buna göre düzeltilmemiştir.

## 4. Metin uzunluğu ve morfoloji

| Sözcük sayısı | Hata oranı | n |
|---|---:|---:|
| ≤ 8 | 0,058 | 103 |
| 9–10 | 0,108 | 1.354 |
| 11–12 | 0,107 | 1.678 |
| ≥ 13 | 0,109 | 128 |

Test setinde uzunluk belirleyici değil (tüm metinler kısa). **Gerçekten kısa girdiler** (yalnızca
terim, 1–3 sözcük) ayrı ölçüldü: doğruluk 0,67, macro F1 0,52 — kısa girdi en büyük zayıflık;
güven skorları ayrı sıcaklıkla (T_short) düzeltildi (ECE 0,124→0,070).

Morfoloji / sözlük dışı (OOV): hatalı örneklerde yüzey-biçim OOV oranı 0,144, doğrularda
0,128; F5 köklemeden sonra 0,040 ve 0,030. F5 kök + karakter n-gram birlikte OOV'yi ~%70
azaltıyor; kalan fark küçük → morfoloji ana hata kaynağı değil.

Türkçe büyük/küçük harf: "IŞIK HIZI VE GÖRELİLİK…" ile küçük harfli hali aynı skorları üretiyor
(test: `test_turkish_i_variants_are_normalized_consistently`). Kalan risk: ASCII "Istanbul"
yazımı Türkçe kuralla "ıstanbul"a döner ve "İstanbul" ile eşleşmez.

## 5. Kritik kuantum ayrımı

| Metin | Sonuç | Güven |
|---|---|---:|
| Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir. | Fizik > Kuantum Mekaniği | 1,00 |
| Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir. | Teknoloji > Kuantum Bilgisayarlar | 0,999 |
| "Kuantum parçacık dalga dolanıklık" / "Kuantum kübit işlemci algoritma" | Fizik / Teknoloji (en olası) | – |

Model "kuantum" sözcüğüne değil bağlama bakıyor: "kübit" sözcüğünün Teknoloji ağırlığı 1,61,
"kuantum"un 0,44 (`TopicClassifier.keywords`). Sınırlama: Kuantum Bilgisayarlar alt konusu
yalnızca 32 eğitim kartına dayanıyor; "Kuantum işlemciler algoritmaları hızlandırır." (kübit
geçmeyen) cümlesi Teknoloji'ye 0,55 güvenle, yani "Belirsiz" olarak gidiyor.

## 6. Domain dışı ve belirsiz girdiler

* Eğitimde hiç görülmemiş 27 kategori (6.060 metin): %69,3 "Diğer", %18,9 "Belirsiz",
  **%11,8 yanlışlıkla bir konuya atandı** (ortalama güven 0,92). En çok sızan kategoriler:
  demircilik %40,6 (→ Kimya/metalurji), balıkçılık %36,6 (→ Biyoloji), Türkiye coğrafyası %31,6
  (→ Tarih), dünya coğrafyası %23,3. Bunlar da anlamca komşu alanlar.
* Elle yazılmış OOD probları: "Akşam pizza söyleyeceğim." Diğer (0,97); "Yarın sabah dişçiye
  gitmem gerekiyor." Diğer (0,93); "Bu kazağın rengi bana hiç yakışmadı." Diğer (0,78).
* Selamlaşma: "Merhaba" → Diğer (0,54; kısa girdi).

## 7. Birden fazla konu, ironi

* "Osmanlı döneminde yazılmış astronomi kitapları incelendi." → Tarih 0,78, Kitaplar 0,19.
* "Futbolcuların performansı yapay zeka ile analiz ediliyor." → Spor 0,75, Teknoloji 0,25.
* "Einstein'ın hayatını anlatan biyografi kitabını bitirdim." → Kitaplar 0,94.
* İroni: "Harika, yine bilgisayar çöktü, tam da teslim günü!" → Diğer 0,76 (Teknoloji 0,09).
  Model ironiyi modellemiyor; burada gündelik şikâyet olarak taksonomi dışı sayılması kabul
  edilebilir, ancak genel olarak ironi/argo eğitim verisinde yok.
* Çok konulu metinler tek etiketle eğitildiği için model tek bir baskın konu seçer; ikinci konu
  olasılıklarda görünür ve **sohbet katmanında** birleşik temaya dönüşür.

## 8. Alan/üslup kayması — harici insan yazımı metinler

| Kaynak | Hedef | Recall (argmax) | Recall (eşikli) | Belirsiz | Diğer |
|---|---|---:|---:|---:|---:|
| MEB biyoloji (BQuAD) | Biyoloji | 0,730 | 0,573 | %27,5 | %7,8 |
| Osmanlı tarihi RC | Tarih | 0,660 | 0,445 | %26,8 | %27,8 |

Gerçek hatalar: "Bu çalışmalarından dolayı Watson ve Crick, 1962 yılında Nobel Ödülü aldılar."
→ Diğer (0,96); "Ama Cem Sultan bu teklifi reddetti…" → Diğer (0,97); "Kemosentezde
fotosentezden farklı olarak ışık enerjisi yerine kimyasal enerji kullanılır." → Belirsiz (en
olası Biyoloji, 0,75'in hemen altında). Anlatı cümleleri (kişi/olay) tanım üslubundaki eğitim
verisinden uzak; tarih anlatısında düşüş belirgin. Bu, sistemin gerçek sohbet metinlerindeki
performansının test skorunun altında olacağının en doğrudan kanıtıdır (KNOWN_ISSUES KI-003).

## 9. Kabul testi B

"Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder." → Belirsiz; en olası
Bilim 0,49, Diğer 0,24. Neden: "Bilim" eğitim kartlarının tamamı epistemoloji; "hipotez"
geçen 16 eğitim kartının 13'ü "Diğer" (ekonometri, dilbilim). Sınıf bazlı eşik denendi
(EXPERIMENTS E18), val'de iyileşme olmadığı için uygulanmadı. Test cümlesini veya benzerini
eğitime eklemek §70 gereği yapılmadı. Sohbet katmanı yumuşak olasılıklarla çalıştığı için bu
mesaj yine de temayı "bilimsel kitaplar"a taşıyor (tests/test_integration.py).
