import numpy as np
from fastFM import als, sgd

class FastFMClassifier(als.FMClassification):
    def __init__self(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        classes_ = np.unique(y)
        y = 2 * (classes_[1] == y) - 1
        super().fit(X, y)
        self.classes_ = classes_
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        return np.column_stack((1 - p, p))


class FastFMSgdClassifier(sgd.FMClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        classes_ = np.unique(y)
        y = 2 * (classes_[1] == y) - 1
        super().fit(X, y)
        self.classes_ = classes_
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        return np.column_stack((1-p, p))