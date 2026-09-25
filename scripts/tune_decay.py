"""Sohbet takibindeki decay değerini simüle sohbetlerle karşılaştırır.

Senaryolar test metinlerinden kurulur: A konusundan 4 mesajın ardından B konusundan 4 mesaj
(izleme doğruluğu, geçiş gecikmesi, konu karışımı) ve üç farklı konudan birer mesaj
(üç konunun temada birlikte korunması). Sonuç: reports/decay_experiment.json
"""

import json
import random
import sys

import numpy as np

from src.config import OTHER_LABEL, RANDOM_SEED, REPORTS_DIR
from src.data.build import load_split
from src.models.predictor import TopicClassifier, aggregate
from src.models.taxonomy import GENERAL_TOPICS
from src.services.conversation import ConversationTracker

DECAYS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
N_CONVERSATIONS = 400
PHASE_LEN = 4


def main() -> int:
    clf = TopicClassifier.load()
    test = load_split("test")
    test = test[test.general != OTHER_LABEL].reset_index(drop=True)
    proba = clf.predict_proba_auto(test["text"].tolist())
    outputs = []
    for row in proba:
        general, cond = aggregate(clf.classes, row)
        joint = {g: {s: p * general[g] for s, p in subs.items()} for g, subs in cond.items()}
        outputs.append((general, joint))
    by_topic = {t: np.flatnonzero(test.general.to_numpy() == t) for t in GENERAL_TOPICS}

    rng = random.Random(RANDOM_SEED)
    conversations = []
    for _ in range(N_CONVERSATIONS):
        a, b = rng.sample(list(GENERAL_TOPICS), 2)
        idx = [int(rng.choice(by_topic[a])) for _ in range(PHASE_LEN)] + \
              [int(rng.choice(by_topic[b])) for _ in range(PHASE_LEN)]
        conversations.append((a, b, idx))

    topic_sets = [rng.sample(list(GENERAL_TOPICS), 3) for _ in range(N_CONVERSATIONS)]
    triples = [(t, [int(rng.choice(by_topic[x])) for x in t]) for t in topic_sets]

    results = {}
    for decay in DECAYS:
        retained = []
        for topics, idx in triples:
            tracker = ConversationTracker(decay=decay)
            for i in idx:
                tracker.update(*outputs[i])
            retained.append(set(topics) <= {t for t, _ in tracker.theme().topics})
        hits, lags, blends, flips = [], [], [], []
        for a, b, idx in conversations:
            tracker = ConversationTracker(decay=decay)
            lag = PHASE_LEN + 1
            for turn, i in enumerate(idx):
                tracker.update(*outputs[i])
                theme = tracker.theme()
                top = theme.topics[0][0] if theme.topics else None
                truth = a if turn < PHASE_LEN else b
                hits.append(top == truth)
                if 1 <= turn < PHASE_LEN:
                    flips.append(top != a)
                if turn >= PHASE_LEN and top == b and lag > PHASE_LEN:
                    lag = turn - PHASE_LEN + 1
                if turn == PHASE_LEN:
                    names = {t for t, _ in theme.topics}
                    blends.append(a in names and b in names)
            lags.append(lag)
        results[str(decay)] = {
            "tracking_accuracy": round(float(np.mean(hits)), 4),
            "mean_switch_lag": round(float(np.mean(lags)), 3),
            "blend_at_switch": round(float(np.mean(blends)), 4),
            "spurious_flip_rate": round(float(np.mean(flips)), 4),
            "three_topic_retention": round(float(np.mean(retained)), 4),
        }
        print(decay, results[str(decay)])
    (REPORTS_DIR / "decay_experiment.json").write_text(
        json.dumps({"n_conversations": N_CONVERSATIONS, "phase_len": PHASE_LEN,
                    "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
