import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

class WeightMixtureClassifier(BaggingClassifier):
  """
  引数: sklearn.ensemble.BaggingClassifier と同じ
  ただし, base_estimator は `coef_`, `intercept_` を持つ線形モデルにのみ対応
  Parameter mixing 用オブジェクト. coef_ を作るタイプの推定器をbaggingする際にパラメータの時点で平均化する.
  GLM系のモデルなら理論上同じようにできるはずだが, とりあえず LogisticRegression で機能するように.
  ただしこの実装では Mann et al. (2009) が主張するようなパフォーマンスの恩恵を得られない
  binary classification しか対応していない
  参考:
  Mann, G. S., McDonald, R., Mohri, M., Silberman, N., & Walker, D. (2009). "Efficient large-scale distributed training of conditional maximum entropy models", Advances in neural information processing systems 22 (pp. 1231–1239). https://papers.nips.cc/paper/3881-efficient-large-scale-distributed-training-of-conditional-maximum-entropy-models
  """
  def fit(self, X, y, **kwargs):
    super().fit(X, y, **kwargs)
    if hasattr(self.estimators_[0], 'intercept_'):
      intercept = np.concatenate([est.intercept_ for est in self.estimators_]).mean()
      self.intercept_ = intercept
    weights = np.concatenate([est.coef_ for est in self.estimators_]).mean(axis=0)
    self.coef_ = weights
    return self

  def predict_proba(self, X):
    if hasattr(self, 'intercept_'):
      intercept = self.intercept_
    else:
      intercept = 0.0
    if issparse(X):
      z = X.dot(self.coef_)
    else:
      z = np.inner(X, self.coef_)
    p = expit(z + intercept).reshape(-1, 1)
    return np.concatenate((1-p, p), axis=1)
  
  def predict_log_proba(self, X):
    return np.log(self.predict_proba(X))
  
  def predict(self, X):
    return self.predict_proba(X) >= 0.5