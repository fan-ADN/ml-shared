import numpy as np
from fastFM import als, sgd, mcmc
import warnings
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_indexing
from sklearn.preprocessing import FunctionTransformer


class FastFMALSClassifier(als.FMClassification):
    def fit(self, X, y):
        classes_ = np.unique(y)
        y = 2 * (classes_[1] == y) - 1
        super().fit(X, y)
        self.classes_ = classes_
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        return np.column_stack((1 - p, p))


class FastFMSGDClassifier(sgd.FMClassification):
    def fit(self, X, y):
        classes_ = np.unique(y)
        y = 2 * (classes_[1] == y) - 1
        super().fit(X, y)
        self.classes_ = classes_
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        return np.column_stack((1 - p, p))


class FastFMMCMCClassifier(mcmc.FMClassification):
    def fit(self, X, y):
        classes_ = np.unique(y)
        y = 2 * (classes_[1] == y) - 1
        super().fit(X, y)
        self.classes_ = classes_
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X)
        return np.column_stack((1 - p, p))

