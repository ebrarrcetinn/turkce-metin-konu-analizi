import pytest

from src.config import OTHER_LABEL
from src.services.conversation import ConversationTracker, Theme, compose_phrase
from src.services.query_builder import build_query, query_form


def test_decay_formula_exact():
    t = ConversationTracker(decay=0.7)
    t.update({"Fizik": 0.8, "Diğer": 0.2})
    t.update({"Fizik": 0.5, "Kimya": 0.5})
    assert t.scores["Fizik"] == pytest.approx(0.8 * 0.7 + 0.5)
    assert t.scores["Kimya"] == pytest.approx(0.5)
    assert t.other_score == pytest.approx(0.2 * 0.7)


def test_invalid_decay():
    with pytest.raises(ValueError):
        ConversationTracker(decay=1.0)


def test_reset_clears_context():
    t = ConversationTracker()
    t.update({"Spor": 1.0})
    t.reset()
    assert t.turns == 0 and not t.scores
    assert t.theme().uncertain


def test_theme_uncertain_when_out_of_scope_dominates():
    t = ConversationTracker()
    t.update({OTHER_LABEL: 0.95, "Spor": 0.05})
    assert t.theme().uncertain
    assert t.theme().label == "Belirsiz"


def test_theme_focus_subtopic():
    t = ConversationTracker()
    t.update({"Teknoloji": 0.95, OTHER_LABEL: 0.05},
             {"Teknoloji": {"Kuantum Bilgisayarlar": 0.9, "Donanım": 0.05}})
    theme = t.theme()
    assert theme.focus_subtopic == "Kuantum Bilgisayarlar"
    assert theme.label == "Teknoloji > Kuantum Bilgisayarlar"


def test_conversation_narrowing_books_science_biology():
    """Kitaplar -> Bilim -> Biyoloji sırasında tema giderek daralır."""
    t = ConversationTracker(decay=0.7)
    t.update({"Kitaplar": 0.95, OTHER_LABEL: 0.05})
    assert t.theme().phrase == "kitaplar"
    t.update({"Bilim": 0.85, OTHER_LABEL: 0.15})
    assert t.theme().phrase == "bilimsel kitaplar"
    t.update({"Biyoloji": 0.95, OTHER_LABEL: 0.05})
    assert t.theme().phrase == "biyoloji hakkında bilimsel kitaplar"


@pytest.mark.parametrize("topics, phrase", [
    (["Kitaplar", "Bilim"], "bilimsel kitaplar"),
    (["Bilim", "Kitaplar"], "bilimsel kitaplar"),
    (["Kitaplar", "Tarih"], "tarih kitapları"),
    (["Biyoloji", "Bilim", "Kitaplar"], "biyoloji hakkında bilimsel kitaplar"),
    (["Fizik", "Teknoloji"], "fizik ve teknoloji"),
    (["Kimya", "Bilim"], "kimya alanında bilimsel araştırmalar"),
    (["Spor"], "spor"),
    ([], ""),
])
def test_compose_phrase(topics, phrase):
    assert compose_phrase(topics) == phrase


def _theme(topics, phrase, focus=None):
    return Theme(topics=[(t, 1.0 / len(topics)) for t in topics], phrase=phrase,
                 focus_subtopic=focus)


def test_query_none_when_uncertain():
    assert build_query(Theme(topics=[], phrase="", uncertain=True), ["kübit"]) is None


def test_query_focus_and_keywords_dedup():
    theme = _theme(["Teknoloji"], "teknoloji", "Kuantum Bilgisayarlar")
    assert build_query(theme, ["kübit", "kuantum", "işlemci"]) == \
        "kuantum bilgisayarlar kübit işlemci"


def test_query_removes_filler_and_limits_keywords_for_long_core():
    theme = _theme(["Biyoloji", "Bilim", "Kitaplar"], "biyoloji hakkında bilimsel kitaplar")
    assert build_query(theme, ["dna", "hücreler"]) == "biyoloji bilimsel kitaplar dna"


def test_query_skips_verbs_and_strips_case_suffix():
    theme = _theme(["Spor"], "spor", "Futbol")
    assert build_query(theme, ["kazandı", "penaltıyla", "derbide"]) == "futbol penaltı derbi"


@pytest.mark.parametrize("token, expected", [
    ("okudum", None), ("düşünüyorum", None), ("gidecek", None), ("kedi", "kedi"),
    ("evrimden", "evrim"), ("dna", "dna"), ("bilgisayar", "bilgisayar"),
])
def test_query_form(token, expected):
    assert query_form(token) == expected


def test_query_length_cap():
    theme = _theme(["Spor"], "spor " * 40)
    assert len(build_query(theme, [])) <= 100
