import numpy as np
from fastFM import als, sgd, mcmc
import warnings
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_indexing
from sklearn.preprocessing import FunctionTransformerF

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


class TestFMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, method='als', *args, **kwargs):
        if method == 'als':
            self.estimator = als.FMClassification(*args, **kwargs)
        elif method == 'sgd':
            self.estimator = sgd.FMClassification(*args, **kwargs)
        elif method == 'mcmc':
            self.estimator = mcmc.FMClassification(*args, **kwargs)
        else:
            raise ValueError("`method` must be 'als', 'sgd', or 'mcmc' ")

    def fit(self, X, y):
        self.classes_ = y.unique()
        y = (y == self.classes_[0]) * 1
        self.estimator.fit(X, 2 * y - 1)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)
        return np.column_stack((1 - p, p))