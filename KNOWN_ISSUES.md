# Bilinen Sorunlar

## KI-001 — Kabul testi B ("Bilim") güvenle sınıflandırılamıyor
* **Severity:** Orta
* **Cause:** "Bilim" sınıfının tek kaynağı `epistemoloji` kartları (249 örnek, bilgi felsefesi).
  "Hipotez" geçen 16 eğitim kartının 13'ü "Diğer"de (ekonometri, dilbilim).
* **Impact:** "Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder." → Belirsiz
  (en olası Bilim, 0,49 < 0,75). Test: `test_acceptance_b_science_confident` (xfail, strict).
  Sohbet teması yine "bilimsel kitaplar"a ilerliyor (yumuşak olasılık).
* **Attempted Solution:** Sınıf bazlı güven eşikleri (EXPERIMENTS E18) → validation'da iyileşme
  yok, reddedildi. Ekonometriyi "Bilim"e eşlemek anlamca yanlış bulundu. Test cümlesini/benzerini
  eğitime eklemek §70 gereği yapılmadı.
* **Recommended Solution:** Bilimsel yöntem/araştırma/bilim tarihi içeren açık lisanslı insan
  yazımı veri eklemek (ör. lisans uygun Türkçe Vikipedi "Bilim" kategori makaleleri, CC BY-SA)
  ve modeli yeniden eğitmek.

## KI-002 — Canlı web araması bu ortamda test edilemedi
* **Severity:** Orta
* **Cause:** Geliştirme ortamının ağ politikası tr.wikipedia.org ve api.duckduckgo.com'u
  engelliyor (proxy 403).
* **Impact:** Ayrıştırıcılar API dokümantasyonuna göre hazırlanmış örnek yanıtlarla
  (`tests/fixtures/`), istemci sahte HTTP oturumuyla test edildi; gerçek API ile uçtan uca
  çağrı **çalıştırılmadı**. Çevrimdışı davranış gerçek ağ hatasıyla doğrulandı.
* **Recommended Solution:** İnternetli bir makinede
  `NLP_NETWORK_TESTS=1 python -m pytest -m network` çalıştırmak.

## KI-003 — Alan/üslup kayması
* **Severity:** Orta
* **Cause:** Eğitim metinleri 5–13 sözcüklük sözlük tanımları.
* **Impact:** İnsan yazımı ders kitabı/tarih anlatısı metinlerinde recall 0,73 / 0,66 (argmax);
  eşikli 0,57 / 0,45. Gerçek sohbet mesajlarında test skorundan daha düşük başarı beklenmeli.
* **Recommended Solution:** Sohbet/anlatı üslubunda etiketli veri; ya da önceden eğitilmiş
  Türkçe cümle gömmeleri (ağ erişimi olan ortamda).

## KI-004 — Transformer karşılaştırması yapılamadı
* **Severity:** Düşük (sonucu değiştirebilir ama mevcut sistem çalışıyor)
* **Cause:** huggingface.co erişimi engelli.
* **Recommended Solution:** BERTurk (dbmdz/bert-base-turkish-cased) ince ayarı ve
  sentence-embedding + LR adaylarını aynı bölmelerle `scripts/run_experiments.py`'ye eklemek.

## KI-005 — Kaynak veride terim–tanım uyuşmazlığı
* **Severity:** Düşük–Orta
* **Cause:** Kaynak veri; 60 kartlık elle incelemede ~%28 uyuşmazlık, ~%27 kesik tanım.
* **Impact:** Kategori düzeyinde etiket büyük ölçüde doğru; ancak bazı test "hataları" gürültü
  kaynaklı (ERROR_ANALYSIS §3).
* **Recommended Solution:** Kaynak projeye geri bildirim; model tabanlı tutarlılık filtresi.

## KI-006 — Kısa ve ASCII Türkçe girdiler
* **Severity:** Düşük
* **Impact:** 1–3 sözcüklük girdilerde doğruluk ~0,67 (güven skorları buna göre kalibre).
  "Istanbul", "cok" gibi ASCII yazımlar Türkçe küçük harf kuralıyla farklı belirteçlere dönüşür.
* **Recommended Solution:** ASCII→Türkçe karakter geri kazanımı (deasciifier) ön işlemesi.
