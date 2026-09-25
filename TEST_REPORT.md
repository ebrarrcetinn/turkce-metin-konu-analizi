# Test Raporu

Bu dosya `python -m scripts.make_test_report` tarafından **2026-09-25T07:15:17+00:00** tarihinde, testler gerçekten çalıştırılarak üretildi; elle düzenlenmedi.

Ortam: Python 3.11.15, Linux-6.18.44-fc-v37-x86_64-with-glibc2.39.

## Özet

| Kontrol | Komut | Sonuç |
|---|---|---|
| Birim + entegrasyon + kabul testleri | `python -m pytest` | 140 passed, 1 skipped, 1 xfailed in 6.78s (çıkış kodu 0) |
| Lint | `ruff check .` | temiz (çıkış kodu 0) |
| Tip denetimi | `mypy src scripts proje.py train.py evaluate.py` | Success: no issues found in 37 source files (çıkış kodu 0) |

Sayım: PASS 140, FAIL 0, SKIP 1, XFAIL (bilinen sorun) 1.

* SKIP: canlı ağ testi (`NLP_NETWORK_TESTS=1` gerekli) — bu ortamda Wikipedia erişimi engelli olduğu için **çalıştırılmadı** (KNOWN_ISSUES KI-002).
* XFAIL: kabul testi B'nin güvenli sınıflandırması (KNOWN_ISSUES KI-001). `strict=True`: test beklenmedik şekilde geçerse paket başarısız olur.

## Kritik ve kabul testleri (model çıktıları, `reports/evaluation.json`)

| Test | Beklenen | Gerçekleşen | Güven | Pass/Fail |
|---|---|---|---:|---|
| Kuantum: Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir. | Fizik > Kuantum Mekaniği | Fizik > Kuantum Mekaniği | 1.000 | PASS |
| Kuantum: Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir. | Teknoloji > Kuantum Bilgisayarlar | Teknoloji > Kuantum Bilgisayarlar | 0.999 | PASS |
| Kabul: Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı. | Kitaplar | Kitaplar > Edebiyat, Roman ve Şiir | 1.000 | PASS |
| Kabul: Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder. | Bilim | Belirsiz | 0.491 | FAIL |
| Kabul: Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir. | Biyoloji | Biyoloji > Genetik | 0.983 | PASS |

OOD (eğitimde görülmemiş 27 kategori, n=6060): reddetme oranı 0.882, yanlış kabul 0.118.

## Tüm testler

| Modül | Test | Beklenen | Gerçekleşen | Pass/Fail |
|---|---|---|---|---|
| test_calibration_and_artifact | `test_fit_temperature_recovers_true_temperature` | fit temperature recovers true temperature | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_temperature_classifier_keeps_argmax_and_normalizes` | temperature classifier keeps argmax and normalizes | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_ece_bounds` | ece bounds | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_multilabel_set_metrics` | multilabel set metrics | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_artifact_roundtrip_and_tamper_detection` | artifact roundtrip and tamper detection | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_artifact_missing_gives_instructions` | artifact missing gives instructions | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_artifact_bad_metadata` | artifact bad metadata | beklenen davranış gözlendi | PASS |
| test_calibration_and_artifact | `test_artifact_class_mismatch` | artifact class mismatch | beklenen davranış gözlendi | PASS |
| test_config | `test_invalid_env_float_falls_back` | invalid env float falls back | beklenen davranış gözlendi | PASS |
| test_config | `test_threshold_grids` | threshold grids | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_decay_formula_exact` | decay formula exact | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_invalid_decay` | invalid decay | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_reset_clears_context` | reset clears context | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_theme_uncertain_when_out_of_scope_dominates` | theme uncertain when out of scope dominates | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_theme_focus_subtopic` | theme focus subtopic | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_conversation_narrowing_books_science_biology` | Kitaplar -> Bilim -> Biyoloji sırasında tema giderek daralır. | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics0-bilimsel kitaplar]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics1-bilimsel kitaplar]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics2-tarih kitaplar\u0131]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics3-biyoloji hakk\u0131nda bilimsel kitaplar]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics4-fizik ve teknoloji]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics5-kimya alan\u0131nda bilimsel ara\u015ft\u0131rmalar]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics6-spor]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_compose_phrase[topics7-]` | compose phrase | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_none_when_uncertain` | query none when uncertain | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_focus_and_keywords_dedup` | query focus and keywords dedup | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_removes_filler_and_limits_keywords_for_long_core` | query removes filler and limits keywords for long core | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_skips_verbs_and_strips_case_suffix` | query skips verbs and strips case suffix | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[okudum-None]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[d\xfc\u015f\xfcn\xfcyorum-None]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[gidecek-None]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[kedi-kedi]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[evrimden-evrim]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[dna-dna]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_form[bilgisayar-bilgisayar]` | query form | beklenen davranış gözlendi | PASS |
| test_conversation_and_query | `test_query_length_cap` | query length cap | beklenen davranış gözlendi | PASS |
| test_database | `test_schema_created_with_version` | schema created with version | beklenen davranış gözlendi | PASS |
| test_database | `test_migration_is_idempotent` | migration is idempotent | beklenen davranış gözlendi | PASS |
| test_database | `test_newer_schema_is_rejected` | newer schema is rejected | beklenen davranış gözlendi | PASS |
| test_database | `test_insert_and_read_history` | insert and read history | beklenen davranış gözlendi | PASS |
| test_database | `test_foreign_keys_enforced` | foreign keys enforced | beklenen davranış gözlendi | PASS |
| test_database | `test_sql_injection_is_stored_literally` | sql injection is stored literally | beklenen davranış gözlendi | PASS |
| test_database | `test_count_rejects_unknown_table` | count rejects unknown table | beklenen davranış gözlendi | PASS |
| test_database | `test_cache_ttl` | cache ttl | beklenen davranış gözlendi | PASS |
| test_database | `test_session_end_and_duplicate_message_index` | session end and duplicate message index | beklenen davranış gözlendi | PASS |
| test_database | `test_write_failure_raises_database_error` | write failure raises database error | beklenen davranış gözlendi | PASS |
| test_database | `test_unopenable_path_raises` | unopenable path raises | beklenen davranış gözlendi | PASS |
| test_database | `test_read_failure_raises_database_error` | read failure raises database error | beklenen davranış gözlendi | PASS |
| test_integration | `test_full_chain_persists_everything` | full chain persists everything | beklenen davranış gözlendi | PASS |
| test_integration | `test_real_http_client_through_service_with_fake_transport` | real http client through service with fake transport | beklenen davranış gözlendi | PASS |
| test_integration | `test_search_cache_avoids_repeat_requests` | search cache avoids repeat requests | beklenen davranış gözlendi | PASS |
| test_integration | `test_offline_does_not_break_classification_and_circuit_breaks` | offline does not break classification and circuit breaks | beklenen davranış gözlendi | PASS |
| test_integration | `test_database_failure_does_not_break_classification` | database failure does not break classification | beklenen davranış gözlendi | PASS |
| test_integration | `test_cache_read_failure_falls_back_to_web` | cache read failure falls back to web | beklenen davranış gözlendi | PASS |
| test_integration | `test_no_database_mode` | no database mode | beklenen davranış gözlendi | PASS |
| test_integration | `test_empty_message_is_not_tracked_or_saved` | empty message is not tracked or saved | beklenen davranış gözlendi | PASS |
| test_integration | `test_reset_keeps_db_rows_and_changes_session` | reset keeps db rows and changes session | beklenen davranış gözlendi | PASS |
| test_integration | `test_conversation_acceptance_books_science_biology` | Sohbet kabul testi: Kitaplar -> Bilim -> Biyoloji. | beklenen davranış gözlendi | PASS |
| test_integration | `test_conversation_with_spec_sentences_keeps_books_and_biology` | Şartnamedeki A, B, C cümleleriyle: B 'Belirsiz' olsa da yumuşak olasılıkla bağlama | beklenen davranış gözlendi | PASS |
| test_integration | `test_cli_session_commands` | cli session commands | beklenen davranış gözlendi | PASS |
| test_integration | `test_cli_handles_eof` | cli handles eof | beklenen davranış gözlendi | PASS |
| test_integration | `test_cli_missing_model_gives_guidance` | cli missing model gives guidance | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_quantum_disambiguation[Kuantum dolan\u0131kl\u0131kta par\xe7ac\u0131klar\u0131n dalga fonksiyonlar\u0131 birlikte de\u011fi\u015febilir.-Fizik-Kuantum Mekani\u011fi]` | quantum disambiguation | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_quantum_disambiguation[Kuantum i\u015flemciler k\xfcbitler kullanarak belirli algoritmalar\u0131 h\u0131zland\u0131rabilir.-Teknoloji-Kuantum Bilgisayarlar]` | quantum disambiguation | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_quantum_not_keyword_only` | 'kuantum' sözcüğü ortakken ayırt edici bağlam sözcükleri kararı belirlemeli. | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_acceptance_a_books` | acceptance a books | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_acceptance_b_science_top_guess` | B'nin en olası konusu Bilim olmalı (argmax düzeyinde geçer). | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_acceptance_b_science_confident` | acceptance b science confident | Bilinen sorun KI-001: Bilim sınıfı yalnızca epistemoloji kartlarından besleniyor; güven 0.49 < eşik 0.75 olduğundan 'Belirsiz'. | XFAIL (bilinen) |
| test_model_behavior | `test_acceptance_c_biology` | acceptance c biology | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_out_of_domain_not_confidently_assigned[Ak\u015fam pizza s\xf6yleyece\u011fim.]` | out of domain not confidently assigned | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_out_of_domain_not_confidently_assigned[Yar\u0131n sabah di\u015f\xe7iye gitmem gerekiyor.]` | out of domain not confidently assigned | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_out_of_domain_not_confidently_assigned[Hafta sonu anneannemin bah\xe7esinde domates toplad\u0131k.]` | out of domain not confidently assigned | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_empty_like_inputs[]` | empty like inputs | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_empty_like_inputs[   ]` | empty like inputs | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_empty_like_inputs[ve bu da bir]` | empty like inputs | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_empty_like_inputs[!!!]` | empty like inputs | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_empty_like_inputs[12345]` | empty like inputs | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_greeting_is_not_a_topic` | greeting is not a topic | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_very_long_text_is_truncated_not_crashing` | very long text is truncated not crashing | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_emoji_text` | emoji text | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_url_text` | url text | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_turkish_i_variants_are_normalized_consistently` | turkish i variants are normalized consistently | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_multi_topic_text_exposes_several_topics` | multi topic text exposes several topics | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_hierarchy_consistency_and_probabilities` | hierarchy consistency and probabilities | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_confidence_bounds_and_latency` | confidence bounds and latency | beklenen davranış gözlendi | PASS |
| test_model_behavior | `test_keywords_come_from_model_weights` | keywords come from model weights | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_joint_label_roundtrip` | joint label roundtrip | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_aggregate_marginals_and_conditionals` | aggregate marginals and conditionals | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_decide_statuses` | decide statuses | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_decide_per_class_thresholds` | decide per class thresholds | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_select_subtopics_threshold_and_limit` | select subtopics threshold and limit | beklenen davranış gözlendi | PASS |
| test_predictor_logic | `test_prediction_json_serialization` | prediction json serialization | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[I-\u0131]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[\u0130-i]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[I\u011eDIR-\u0131\u011fd\u0131r]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[\u0130STANBUL-istanbul]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[ISPARTA ve \u0130ZM\u0130R-\u0131sparta ve izmir]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_turkish_lower[\xc7\u011e\xd6\u015e\xdc-\xe7\u011f\xf6\u015f\xfc]` | turkish lower | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_python_lower_is_wrong_for_turkish_but_ours_is_not` | python lower is wrong for turkish but ours is not | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_normalize_removes_url_mention_html_emoji_digits` | normalize removes url mention html emoji digits | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_normalize_apostrophe_suffix_and_punctuation` | normalize apostrophe suffix and punctuation | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_normalize_nfc_equivalence` | normalize nfc equivalence | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_fix_mojibake` | fix mojibake | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_normalize_rejects_non_string` | normalize rejects non string | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_tokenize_and_stopwords` | tokenize and stopwords | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[   ]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[ve bu da bir]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[!!!]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[12345]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[... 42 ...]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[\U0001f60d\U0001f525]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_not_meaningful[@kullanici https://x.com]` | not meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_meaningful` | meaningful | beklenen davranış gözlendi | PASS |
| test_preprocessing | `test_f5` | f5 | beklenen davranış gözlendi | PASS |
| test_taxonomy | `test_quantum_cards_split_by_term` | quantum cards split by term | beklenen davranış gözlendi | PASS |
| test_taxonomy | `test_excluded_and_other` | excluded and other | beklenen davranış gözlendi | PASS |
| test_taxonomy | `test_mapping_tables_are_disjoint_and_complete` | mapping tables are disjoint and complete | beklenen davranış gözlendi | PASS |
| test_taxonomy | `test_every_general_topic_has_subtopics_and_hierarchy_is_valid` | every general topic has subtopics and hierarchy is valid | beklenen davranış gözlendi | PASS |
| test_web_search | `test_parse_wikipedia_orders_cleans_and_filters` | parse wikipedia orders cleans and filters | beklenen davranış gözlendi | PASS |
| test_web_search | `test_parse_wikipedia_limit_and_empty_and_error` | parse wikipedia limit and empty and error | beklenen davranış gözlendi | PASS |
| test_web_search | `test_parse_duckduckgo_nested_topics_and_https_only` | parse duckduckgo nested topics and https only | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[https://tr.wikipedia.org/wiki/X-True]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[https://en.wikipedia.org/wiki/X-True]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[https://duckduckgo.com/X-True]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[http://tr.wikipedia.org/wiki/X-False]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[https://tr.wikipedia.org.evil.example/wiki/X-False]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[javascript:alert(1)-False]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[https://evilwikipedia.org/-False]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_url_validation[None-False]` | url validation | beklenen davranış gözlendi | PASS |
| test_web_search | `test_clean_text_unescapes_strips_and_truncates` | clean text unescapes strips and truncates | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_success_uses_timeout_and_utf8` | client success uses timeout and utf8 | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_retries_on_503_then_succeeds` | client retries on 503 then succeeds | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_honors_retry_after_with_cap` | client honors retry after with cap | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_malformed_json_falls_back_to_duckduckgo` | client malformed json falls back to duckduckgo | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_offline` | client offline | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_timeout_is_offline` | client timeout is offline | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_http_404_is_error_not_crash` | client http 404 is error not crash | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_no_results_is_empty` | client no results is empty | beklenen davranış gözlendi | PASS |
| test_web_search | `test_client_other_request_errors_are_contained` | client other request errors are contained | beklenen davranış gözlendi | PASS |
| test_web_search | `test_empty_query` | empty query | beklenen davranış gözlendi | PASS |
| test_web_search | `test_live_wikipedia` | live wikipedia | NLP_NETWORK_TESTS=1 değil | SKIP |
