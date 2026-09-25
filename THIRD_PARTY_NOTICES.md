# Üçüncü Taraf Bildirimleri

Bu proje aşağıdaki veri setlerini ve kütüphaneleri kullanır. Ham veri repoda **dağıtılmaz**;
`scripts/download_data.py` her kullanıcının kendi kopyasını kaynağından indirir.

## Veri setleri

| Veri seti | Kaynak | Lisans | Kullanım |
|---|---|---|---|
| Türkçe Tabu Veri Seti | https://github.com/wleeaf/taboo-dataset (commit `f621b46`) | MIT, Copyright (c) 2026 | Eğitim / değerlendirme |
| Turkish BQuAD | https://github.com/TurQuest/turkish-bquad (commit `30a3e07`) | MIT, Copyright (c) 2021 TurQuest | Harici değerlendirme |
| Turkish Reading Comprehension QA (Osmanlı Tarihi) | https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset (commit `c6852f8`) | MIT, Copyright (c) 2020 Okan | Harici değerlendirme |

MIT lisansı, yazılımın/verinin kopyalarında telif ve izin bildiriminin korunmasını ister.
Kaydedilen model (`models/topic_model.joblib`) tabu verisinden türetilmiş öznitelik sözlükleri
(n-gram'lar) içerir; modeli dağıtırsanız yukarıdaki bildirimi de ekleyin.

## Yazılım bağımlılıkları (requirements.txt)

| Paket | Lisans |
|---|---|
| numpy, scipy, pandas, scikit-learn, joblib | BSD-3-Clause |
| matplotlib | Matplotlib License (PSF tabanlı, BSD uyumlu) |
| requests | Apache-2.0 |

Geliştirme (requirements-dev.txt): pytest (MIT), ruff (MIT), mypy (MIT), types-requests
(Apache-2.0), fasttext-wheel (MIT; yalnızca E12 deneyi).

## Önceden eğitilmiş model

Kullanılmadı. Final model yalnızca yukarıdaki MIT veri seti üzerinde sıfırdan eğitildi.

## Harici servisler (çalışma zamanı)

* Wikipedia API (https://tr.wikipedia.org/w/api.php) — içerik CC BY-SA 4.0; sonuç özetleri
  kaynak URL'siyle birlikte gösterilir ve saklanır.
* DuckDuckGo Instant Answer API (https://api.duckduckgo.com/) — yedek; API anahtarı gerektirmez.
