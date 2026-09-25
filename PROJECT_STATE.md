# Proje Durumu

Son güncelleme: 2026-09-25

## Completed

| Faz | Kapsam (şartname maddeleri) | Kontrol noktası |
|---|---|---|
| 0–2 | Repo incelemesi (boş repo), gereksinim çıkarımı, eksik analizi | – |
| 3–8 | Veri araştırması, seçim, EDA, temizlik, taksonomi, gruplu bölme (§1–§13) | KN-1 ✔ |
| 9–15 | Baseline, gelişmiş modeller, karşılaştırma, tuning, hiyerarşi, kalibrasyon, OOD eşiği, hata analizi, persistence (§14–§20, §29–§30) | KN-1 ✔ (temiz kopyada metrikler birebir aynı) |
| 16–20 | Sohbet takibi + DECAY deneyi, tema, sorgu, web arama, SQLite, CLI (§21–§35) | KN-2 ✔ |
| 21–23 | Unit, integration, acceptance, edge testleri (§36–§40) | KN-2 ✔ (135/1 skip/1 xfail) |
| 24–26 | Performans, güvenlik (pip-audit), gizlilik, dokümanlar, cache, versioning (§41–§60) | KN-3 ✔ |
| 27–28 | Migration, artifact doğrulama, lisans, doküman-kod denetimi, son code review, final rapor (§61–§87) | KN-4 (final) ✔ |

## In Progress

Yok.

## Decisions

DECISIONS.md (ADR-001 … ADR-015). Özet: tabu veri seti; terim-gruplu bölme; E11 (F5 word + char
TF-IDF + LinearSVC); düz birleşik etiket; temperature scaling; eşik 0,75; T_short 0,208;
DECAY 0,7; rol tabanlı tema; model ağırlıklı sorgu; Wikipedia→DDG + devre kesici.

## Metrics

| | Değer |
|---|---:|
| Val macro F1 (eşikli) | 0,849 |
| Test macro F1 (eşikli) / accuracy / weighted F1 | 0,850 / 0,894 / 0,893 |
| Test ECE | 0,037 |
| OOD reddetme | 0,882 |
| Harici recall (Biyoloji / Tarih, argmax) | 0,730 / 0,660 |
| Tek metin çıkarım p50 | 2,2 ms |
| Testler | 140 geçti, 1 atlandı, 1 xfail (son kod incelemesinden sonra) |

## Known Problems

KNOWN_ISSUES.md: KI-001 (Bilim/kabul B), KI-002 (canlı web testi yok), KI-003 (üslup kayması),
KI-004 (Transformer yok), KI-005 (kaynak veri gürültüsü), KI-006 (kısa/ASCII girdi).

## Next Actions

1. İnternetli makinede `NLP_NETWORK_TESTS=1 python -m pytest -m network`.
2. Hugging Face erişimli ortamda BERTurk karşılaştırması (KI-004).
3. Bilim sınıfı için ek veri (KI-001).

## Definition of Done (§84)

| Madde | Durum |
|---|---|
| Dataset araştırıldı; Kaggle/UCI adayları karşılaştırıldı; seçim gerekçelendirildi | ✔ (Kaggle/UCI/HF ortamdan erişilemedi → dokümantasyonla değerlendirildi) |
| Lisans kontrolü; data provenance | ✔ (MIT; commit + SHA-256 + provenance.json) |
| EDA; duplicate; leakage; class imbalance | ✔ |
| Train/val/test düzgün ayrıldı | ✔ (terim-gruplu, stratified) |
| Baseline; alternatif modeller; karşılaştırma; deneye dayalı final seçim | ✔ (12 aday; Transformer ölçülemedi — KI-004) |
| Hyperparameter seçimi gerekçeli; macro F1; per-class; confusion matrix | ✔ |
| Error analysis; confidence davranışı; OOD | ✔ |
| Genel konu ve alt konu sınıflandırması | ✔ |
| Multi-label gerekiyorsa doğru uygulandı | ✔ (veri tek etiketli; eşik 0,20–0,70 tarandı, tek etiket en iyi) |
| Sohbet context'i; topic narrowing; query generation | ✔ |
| Web search çalışıyor | ◐ Uygulandı ve sahte HTTP ile test edildi; **canlı API bu ortamda doğrulanamadı** (KI-002) |
| Offline durumda çökmeme | ✔ (gerçek ağ hatasıyla doğrulandı) |
| SQLite kayıtları; yeniden eğitmeden yükleme | ✔ |
| Unit / integration testleri; acceptance ve edge-case testleri çalıştırıldı | ✔ (kabul 2/3 geçti; B bilinen sorun) |
| Performance; security review | ✔ |
| README, ARCHITECTURE, DATASET_CARD, MODEL_CARD, EXPERIMENTS, ERROR_ANALYSIS, TEST_REPORT, FINAL_REPORT, PROJECT_STATE | ✔ |
| requirements.txt doğru; kod-doküman uyumu | ✔ (`scripts/check_docs.py` 0 sorun) |
| Temiz ortamdan kurulup çalıştırılabiliyor | ✔ (yeni venv + git clone, KN-3 ve KN-4) |
| Kritik yapılacak/düzeltilecek notu kalmadı | ✔ (kod ve dokümanlarda tarandı) |
