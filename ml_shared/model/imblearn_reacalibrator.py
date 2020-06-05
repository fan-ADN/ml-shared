import numpy as np
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from ..evaluation import calibrate_imbalanceness
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

def get_oversampling_rate(post_positive_rate):
    """
    imblearn.over_sampler.RandomOversamplerで任意の比率でサンプリングするための関数
    このクラスの設定が分かりにくいので作った
    :param post_positive_rate: リサンプリング後の少数例の割合. 例: 1%しかないのを全体の30%にしたいなら 0.3, 均等にしたいなら0.5
    :return:
    """
    return post_positive_rate / (1 - post_positive_rate)


def get_oversampling_power(post_positive_rate, pos_rate):
    """
    calibrate_imbalancenes の pos_rateに与えるやつ
    :param post_positive_rate: サンプリング後の少数例の割合
    :param pos_rate: リサンプリング前の少数例の割合
    :return:
    """
    return get_oversampling_rate(post_positive_rate) * (1 - pos_rate) / pos_rate

class imblearn_recalibrator(BaseEstimator, ClassifierMixin):
    """
    imblearnのリサンプリングの偏りを再較正するやつ
    再較正のコードを毎回書きたくない. scikit-learnの設計思想に則りオブジェクト指向プログラミングをしよう
    estimator, resampler, サンプリング割合を指定したら後は fit & predict/predict_proba するだけ
    * 注意: 不均衡データに対するリサンプリングは分類性能を目的としているので判別性能等に効果があるかは知らない
    
    :param estimatror: scikit-learn API 準拠の estimator オブジェクト
    :param resampler: imblearn で使われる各種 resampler オブジェクト
    :param post_minor_rate: リサンプリング後の**全件に対する少数例の割合**を指定. default is None. alpha とどちらか片方を使う.
    :param alpha: **リサンプリング前に対する**事後の少数例の割合**を指定. default is 'auto'. post_minor_rate とどちらか片方を使う.
    """
    def __init__(self, estimator, resampler, alpha='auto', post_minor_rate=None):
        resampler = clone(resampler)
        if post_minor_rate is None and alpha is None:
            warnings.warn('neither of `post_minor_rate` nor `alpha` are specified. Instead resampling stragegy specified in `resampler` object is used.')
        elif post_minor_rate and alpha: 
            warnings.warn('both of `post_minor_rate` and `alpha` are specified. the former is applied.')
            self.post_minor_rate = post_minor_rate
            self.resampling_strategy = 'posterior_rate'
        elif post_minor_rate:
            self.post_minor_rate = post_minor_rate
            self.resampling_strategy = 'posterior_rate'
        elif alpha:
            self.alpha = alpha
            self.resampling_strategy = 'alpha'
            resampler.set_params(sampling_strategy = alpha)
        else:
            raise('initialized error')
        self.estimator_ = Pipeline([
            ('resampler', resampler),
            ('estimator', clone(estimator))
        ])
    
    def fit(self, X, y):
        if self.resampling_strategy == 'posterior_rate':
            alpha = get_oversampling_rate(self.post_minor_rate)
            self.alpha = alpha
            self.estimator_['resampler'].set_params(sampling_strategy=alpha)
        self.estimator_.fit(X, y)
        self.minor_rate_ = np.min([y.mean(), 1 - y.mean()])
        return self
    
    def predict(self, X):
        return self.estimator_.predict(X)
    def predict_proba(self, X):
        return calibrate_imbalanceness(self.estimator_.predict_proba(X),
         pos_rate=get_oversampling_power(self.alpha, self.minor_rate_))
