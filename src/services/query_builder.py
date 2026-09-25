"""
Dosya   : src/services/query_builder.py
Konu    : Arama Sorgusu Oluşturma
Açıklama: Bu dosyada amacım sohbet temasından ve mesajdaki ayırt edici sözcüklerden
          internet araması için kısa bir sorgu üretmek.
Yazar   : Ebrar Cemre Çetin
Tarih   : 26.09.2026
"""

from __future__ import annotations

import re

from src.preprocessing.text import f5_stem, normalize
from src.services.conversation import Theme

# Tema cümlesindeki bağlaç ve dolgu sözcükleri arama motoruna katkı vermediği için çıkarıyorum.
FILLER_WORDS = frozenset({"alanında", "ve"})
MAX_QUERY_KEYWORDS = 2  # temaya ekleyeceğim en fazla anahtar sözcük
LONG_CORE_WORDS = 3  # bu uzunluktaki tema ifadesi zaten belirgin; sözcük eklemiyorum
MAX_QUERY_CHARS = 100  # çok uzun sorgular arama motorlarında sonuç vermediği için sınırladım
MIN_STEM_CHARS = 4  # eki attıktan sonra bundan kısa kalan sözcüğü eski haliyle bırakıyorum

# Çekimli fiil sonları; bu sözcükleri sorguya eklemiyorum. Geçmiş zaman eki ünsüz uyumuna
# göre yalnızca sert ünsüzden (ç f h k p s ş t) sonra "-tı" olur; böylece "kuantum"
# gibi isimleri fiil sanmıyorum.
FINITE_VERB_RE = re.compile(
    r"(?:[ıiuü]yor(?:um|sun|uz|lar)?|(?:d|(?<=[çfhkpsşt])t)[ıiuü](?:m|n|k|nız|niz|lar|ler)?|"
    r"[mn][ıiuü]ş(?:[ıiuü]m|lar|ler)?|[ae]c[ae]k(?:[ıi]m|lar|ler)?)$"
)
# Sorgu için kırptığım yaygın hal ekleri: -la/-le/-yla (araç), -da/-de/-ta/-te (bulunma),
# -dan/-den/-tan/-ten (ayrılma). "penaltıyla" -> "penaltı", "derbide" -> "derbi"
CASE_SUFFIX_RE = re.compile(r"(?:y?l[ae]|[dt][ae]n|[dt][ae])$")


def query_form(token: str) -> str | None:
    """Anahtar sözcüğü sorgu biçimine getiriyorum; çekimli fiilse None döndürüyorum."""
    # Kısa isimler ("kedi", "vadi") fiil kalıbına yanlışlıkla uyduğu için onları denetlemiyorum.
    if len(token) >= 5 and FINITE_VERB_RE.search(token):
        return None
    stripped = CASE_SUFFIX_RE.sub("", token)
    return stripped if len(stripped) >= MIN_STEM_CHARS else token


def build_query(theme: Theme, keywords: list[str] | None = None) -> str | None:
    """Amacım sohbet temasından arama sorgusu kurmak.

    Tema belirsizse ``None`` döndürüyorum (arama yapmıyorum).

    Sorgu = tema ifadesi + (tema kısaysa) son mesajdaki en ayırt edici 1-2 sözcük.
    Örnek: tema "kuantum bilgisayarlar", mesajda "kübit" -> "kuantum bilgisayarlar kübit".
    Anahtar sözcükleri `TopicModel.keywords` modelin öğrendiği ağırlıklara göre seçer.
    """
    if theme.uncertain or not theme.topics:
        return None
    # Alt konuya odaklanmış sohbette alt konu adını, değilse tema cümlesini çekirdek alıyorum.
    core = theme.focus_subtopic or theme.phrase
    words = [w for w in normalize(core).split() if w not in FILLER_WORDS]
    stems = {f5_stem(w) for w in words}
    budget = 0 if len(words) >= LONG_CORE_WORDS else MAX_QUERY_KEYWORDS
    for keyword in keywords or []:
        if budget == 0:
            break
        kw = query_form(normalize(keyword))
        # Temada zaten geçen (aynı köklü) sözcükleri tekrar eklemiyorum.
        if not kw or f5_stem(kw) in stems:
            continue
        words.append(kw)
        stems.add(f5_stem(kw))
        budget -= 1
    query = " ".join(words)
    if len(query) > MAX_QUERY_CHARS:
        # Sözcük ortasından kesmemek için son boşluğa kadar kırpıyorum.
        query = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
    return query or None
