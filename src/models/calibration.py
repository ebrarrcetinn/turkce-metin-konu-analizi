"""LinearSVC skorlarından temperature scaling ile kalibre olasılık üretir.

Sıcaklık, eğitim verisinin out-of-fold skorları üzerinde öğrenilir; sınıf sıralaması
değişmediği için kararlar temel modelle aynı kalır.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GroupKFold, StratifiedKFold

T_BOUNDS = (0.01, 10.0)


def fit_temperature(scores: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(t: float) -> float:
        return float(-log_softmax(scores / t, axis=1)[np.arange(len(y_idx)), y_idx].mean())

    return float(minimize_scalar(nll, bounds=T_BOUNDS, method="bounded").x)


class TemperatureScaledClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, cv: int = 3, random_state: int = 42, store_oof: bool = False):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.store_oof = store_oof

    def fit(self, X, y, groups=None):
        X = list(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        if groups is not None:
            splitter = GroupKFold(n_splits=self.cv)
            splits = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                       random_state=self.random_state)
            splits = splitter.split(X, y)
        oof = np.zeros((len(X), len(self.classes_)))
        for tr_idx, va_idx in splits:
            model = clone(self.estimator).fit([X[i] for i in tr_idx], y[tr_idx])
            cols = np.searchsorted(self.classes_, model.classes_)
            oof[np.ix_(va_idx, cols)] = model.decision_function([X[i] for i in va_idx])
        self.temperature_ = fit_temperature(oof, y_idx)
        if self.store_oof:
            # Eşik seçimi için geçici; artifact'a kaydedilmeden önce silinir.
            self.oof_proba_ = softmax(oof / self.temperature_, axis=1)
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def decision_function(self, X):
        return self.estimator_.decision_function(list(X))

    def predict_proba(self, X, temperature: float | None = None):
        t = temperature if temperature is not None else self.temperature_
        return softmax(self.decision_function(X) / t, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
