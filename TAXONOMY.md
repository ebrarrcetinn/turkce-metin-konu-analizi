# Taksonomi ve Eşleme

Kod: `src/models/taxonomy.py`. Başlangıç taksonomisi (proje şartnamesi §13) veri setinin
kategori yapısına zorla uydurulmadı; aşağıdaki eşleme ile uyarlandı.

## Son taksonomi (8 genel konu, 38 alt konu + "Diğer")

| Genel Konu | Alt Konular (kaynak kategori) |
|---|---|
| Fizik | Kuantum Mekaniği (`kuantum`), Termodinamik (`termodinamik`), Optik (`optik`), Akustik (`akustik`), Astrofizik ve Astronomi (`astronomi`) |
| Kimya | Genel Kimya (`kimya`), Metalurji ve Malzeme (`metalurji`) |
| Biyoloji | Genetik (`genetik`), Ekoloji (`ekoloji`), Mikrobiyoloji (`mikrobiyoloji`), Botanik (`botanik`), Zooloji (`zooloji`, `kuslar`, `denizcanlilari`), Evrim ve Paleontoloji (`paleontoloji`) |
| Teknoloji | **Kuantum Bilgisayarlar** (`kuantum` + terim kuralı), Yapay Zeka (`yapayzeka`), Donanım (`donanim`), Yazılım ve Algoritmalar (`algoritmalar`), Siber Güvenlik (`siber`), Robotik (`robotik`), Telekomünikasyon (`telekomunikasyon`), Nanoteknoloji (`nanoteknoloji`) |
| Bilim | Bilim Felsefesi ve Yöntem (`epistemoloji`) |
| Kitaplar | Edebiyat, Roman ve Şiir (`edebiyat`), Türk Halk Edebiyatı (`turkhalkedebiyati`), Masallar (`masallar`), Çizgi Roman (`cizgiroman`) |
| Spor | Futbol, Basketbol, Voleybol, Tenis, Atletizm ve Olimpiyatlar, Motor Sporları, Dövüş Sporları, Doğa ve Binicilik Sporları (`binicilik`, `dagcilik`) |
| Tarih | Osmanlı Tarihi (`osmanli`), Cumhuriyet Tarihi (`cumhuriyettarihi`), Savaşlar ve Askeri Tarih (`savas`), Antik Çağ ve Arkeoloji (`arkeoloji`, `epigrafi`) |
| Diğer | Eşlenmeyen 78 kategori (51'i eğitimde, 27'si yalnızca OOD testinde) |

## Başlangıç taksonomisinden farklar ve gerekçeleri

| Şartname alt konusu | Durum | Gerekçe |
|---|---|---|
| Görelilik, Klasik Mekanik | Yok | Kaynakta ayrı kategori yok; Fizik alt konuları veri kategorilerine göre (Termodinamik, Optik, Akustik) belirlendi |
| Hücre Biyolojisi, Evrim | Mikrobiyoloji, Evrim ve Paleontoloji | Kaynakta hücre biyolojisi kategorisi yok; paleontoloji evrim terimlerini içeriyor |
| Organik Kimya, Kimyasal Tepkimeler, Periyodik Tablo | Genel Kimya | Kaynak `kimya` kategorisi bunları ayırmıyor; ayrı etiket uydurulmadı |
| Bilimsel Yöntem, Bilim Tarihi, Bilimsel Araştırma | Bilim Felsefesi ve Yöntem | Tek kaynak `epistemoloji`; yöntem/araştırma kapsamı zayıf (KI-001) |
| Roman, Bilimsel Kitaplar, Şiir, Yazarlar | Edebiyat, Roman ve Şiir; Halk Edebiyatı; Masallar; Çizgi Roman | "Bilimsel kitaplar" bir konu değil **birleşim**dir; sohbet katmanında Kitaplar + Bilim olarak üretilir |
| Olimpiyatlar | Atletizm ve Olimpiyatlar | `atletizm` olimpik branş terimlerini içeriyor |
| Dünya Savaşları | Savaşlar ve Askeri Tarih | `savas` genel askeri tarih içeriyor |

## Kuantum terim kuralı

`kuantum` kategorisindeki bir kart, **yalnızca terim (`kelime`) alanı** şu kalıba uyarsa
Teknoloji > Kuantum Bilgisayarlar'a taşınır (açıklama metni kurala girmez):
`kübit|qubit|kuantum (bilgisayar|işlemci|kapı|devre|algoritma|bellek|hata|hacmi|hızı|üstünlüğü)|
topolojik (kuantum )?bilgisayar|hadamard kapısı|cnot|pauli kapısı|faz kapısı|yüzey kodu|transmon|
hata düzeltme|bloch küresi|kuantum gürültüsünü`. Sonuç: 32 kart Teknoloji'ye, 212 kart Fizik'te.
Kural yalnızca `kuantum` kategorisine uygulanır; `nanoteknoloji` kategorisindeki birkaç kuantum
bilgisayar kartı Teknoloji > Nanoteknoloji'de kaldı (genel konu yine doğru).

## Dışlanan 31 kategori

`anatomi, astronotik, cerrahi, dalgiclik, dermatoloji, epidemiyoloji, farmakoloji, felsefe,
havacilik, immunoloji, internethayati, jeoloji, jeomorfoloji, kripto, mantik, mekatronik,
meteoroloji, mitoloji, monarsi, noroloji, numizmatik, orman, osinografi, patoloji, pediyatri,
psikiyatri, saglikliyasam, sosyalmedya, tiyatro, vahsidoga, videooyunlari`

Bu kategoriler taksonominin bir konusuyla anlamca örtüşür (ör. `anatomi`→Biyoloji mi Diğer mi?
`havacilik`→Teknoloji mi? `felsefe`→Bilim mi? `mitoloji`/`tiyatro`→Kitaplar mı?). "Diğer" olarak
etiketlemek sistematik etiket gürültüsü, pozitif sınıfa katmak ise taksonomiyi keyfi genişletmek
olurdu; bu yüzden hiçbir bölmede kullanılmadılar. Sınırlama: bu alanlardaki kullanıcı metinleri
için davranış tanımsızdır (genellikle en yakın konuya ya da "Belirsiz"e gider).

## Hiyerarşi

Model tek bir düz sınıflandırıcıdır (sınıflar "Genel > Alt"); genel konu olasılığı alt konu
olasılıklarının toplamıdır. Bu yapı gereği tahmin edilen alt konu her zaman tahmin edilen genel
konuya aittir (`validate_hierarchy`, test: `tests/test_model_behavior.py`). Hiyerarşik
(genel → alt model) alternatifle karşılaştırma: EXPERIMENTS.md E-H1/E-H2.
