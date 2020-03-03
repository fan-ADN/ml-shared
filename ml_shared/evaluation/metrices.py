#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:14:58 2019

@author: s_katagiri
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
        mean_absolute_error, roc_auc_score, roc_curve, log_loss)
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg
from plotnine import (
        ggplot, aes, geom_point, geom_line, geom_segment, geom_histogram, geom_vline,
        theme, theme_classic, labs, scale_color_discrete, coord_equal, facet_wrap)
from itertools import chain


def calibrate_imbalanceness(p, pos_rate=1.0, neg_rate=1.0):
    """
    不均衡サンプリング割合に応じて予測確率 p を修正するやつ
    結局逆数を取れば同じだけどケアレスミス回避のため pos/neg それぞれの引数を用意した
    :param p: array-like.
    :param pos_rate: float. 正例のサンプリング倍率。 サンプリング倍率とはリサンプリング前後で件数が何倍になったか, つまり (事後正例数)/(事前正例数). oversampling なら >1, down なら <1. 0 より大きい値とすること
    :param neg_rate: float. 負例のサンプリング率。同上。
    :return:
    """
    if pos_rate <= 0 or neg_rate <= 0:
        raise ValueError("pos_rate or neg_rate must be greater than zero!")
    return p * neg_rate / (pos_rate - (pos_rate - neg_rate) * p)


def entropy(p):
    """
    NE の分母のやつ. logloss だと y が離散でないとエラーが出るため
    :param p: array-like.
    :return: float.
    """
    p = np.array(p).mean()
    return - (p * np.log(p) + (1 - p) * np.log(1 - p))


def normalized_entropy(label, prob):
    """
    正規化クロスエントロピー(NE)を計算する
    compute the Normalized cross entropy (NE)
    NE:= (log loss)/(mean entropy)
    1基準であり, 値が**小さい**ほど当てはまりが良い.

    :param label: array-like.
    :param prob: array-like.
    :return: float/
    """
    return log_loss(label, prob) / entropy(label)


def relative_information_gain(label, prob):
    """
    相対情報ゲイン(RIG)を計算する
    compute the relative information gain (RIG)
     1 - log-loss/mean-entropy
    ゼロが基準であり, 値が**大きい**ほど当てはまりが良い
    Args:
        label: array-like.
        prob: array-like.

    Returns: float
    """
    return 1 - log_loss(label, prob)/entropy(label)


def negative_relative_information_gain(label, prob):
    """
    負の相対情報ゲイン(RIG)を計算する.
    log-loss/mean-entropy - 1
    = NE - 1
    ゼロが基準であり, 値が**小さい**ほど当てはまりが良い
    Args:
        label:
        prob:

    Returns:

    """
    return log_loss(label, prob)/entropy(label) - 1


def normalized_log_loss(label, prob):
    """
    正規化対数損失 (normalized log loss; NLL) の計算.
    NLL := 1 - (log loss) / (mean entropy)
    ゼロが基準, 値が**大きい**ほど当てはまりが良い
    RIGと同じ

    :param label: array-like.
    :param prob: array-like.
    :return: float

    References:
    Lefortier, Damien, Anthony Truchet, and Maarten de Rijke. 2015. ,
    “Sources of Variability in Large-Scale Machine Learning Systems.,”
    In Machine Learning Systems (NIPS 2015 Workshop). http://learningsys.org/2015/papers.html.
    """
    return 1 - log_loss(label, prob) / entropy(label)


def expected_calibration_error(y_true, y_pred, m=10,
                               strategy='quantile', weight=None, **args):
    """
    期待カリブレーション誤差を計算する関数.
    Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht. (2015),
    “_Obtaining Well Calibrated Probabilities Using Bayesian Binning_.,”
    in Proceedings of the AAAI Conference on Artificial Intelligence, 2901–7, PMID: 25927013
    strategy='quantile': グループ分割の方法. デフォルトでは等間隔ではなく分位数
    m=10: グループ数 (ビン数).
    weight=None: 平均誤差計算時の, グループごとの重み, デフォルトでは必要なし.
    args: その他, sklern.metrics.calibration_curve の引数
    :param y_true:
    :param y_pred:
    :param m:
    :param strategy:
    :param weight:
    :param args:
    :return:
    ------
    TODO: quantile で分割できない例外処理
    """
    y, x = calibration_curve(y_true, y_pred, n_bins=m, strategy=strategy, **args)
    return mean_absolute_error(y, x, sample_weight=weight)


def integrated_calibration_index(y, p, **args):
    """
    Integrated Calibration Index (ICI) 計算用
    y ~ pred で LO(W)ESS 回帰したものをカリブレーション曲線とみなして, MAE で評価する
    (本当にいいのかそれで?)
    TODO: 平滑化パラメータの調整
    -------
    y: 0/1ラベル.
    p: 予測リスク.
    args: その他 lowess に与えられる引数
    -------
    TODO: statsmodels.nonparametric.smoothers_lowess.lowess が遅い
    References:
    P. C. Austin and E. W. Steyerberg (2019),
    “The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models,”
    Statistics in Medicine
    """
    smoothed = lowess(y, p, **args)
    return mean_absolute_error(*smoothed.transpose())


def integrated_calibration_index_mod(y, p):
    """
    local reg 使うバージョン
    TOOD: statsmodels.nonparametric.kernel_regression.KernReg がとても遅い
    """
    ll = KernelReg(endog=y, exog=p, reg_type='ll', var_type='o')
    return mean_absolute_error(y, ll.fit()[0])


def print_metrics_2(y_train, y_test, p_train, p_test):
    """
    分類モデルの事後評価レポート: train/test の2つの結果を比較する.
    """
    print('train/test\nsize: {:.0f}/{:.0f}\npos. label rate: {:.4f}/{:.4f}'.format(
    y_train.shape[0], y_test.shape[0], y_train.mean(), y_test.mean()))
    print('AUC: {0[0]:.4f}/{0[1]:.4f}, logloss: {1[0]:.4f}/{1[1]:.4f}, NE: {2[0]:.4f}/{2[1]:.4f}'.format(
        *[(f(y_train, p_train), f(y_test, p_test)) for f in [roc_auc_score, log_loss, normalized_entropy]]
    ))
    print('naive logloss: {:.4f}/{:.4f}'.format(entropy(y_train), entropy(y_test)))
    print('train click/exp: {:.0f}/{:.0f} = {:.4f}\n test click/exp: {:.0f}/{:.0f} = {:.4f}'.format(
        y_train.sum(), p_train.sum(),
        y_train.sum() / p_train.sum(),
        y_test.sum(), p_test.sum(),
        y_test.sum() / p_test.sum()))


def print_metrics(label_list, pred_list, names=None, metrics=None):
    """
    分類モデルの事後評価レポート. 2つ以上のモデルを比較する.
    -------
    :param: label_list: 正解ラベルリストの配列. [(y1, y2, ...), (y1, y2, ...)]  のようにして与える,  pred_list に対応させる
    :param: pred_list: 予測確率リストの配列. label_list と同じ長さにすること
    :param: names=None: モデルの名称. None または同じ長さにすること
    :param: metrics=None: 計算する統計量のリスト. リスト要素は (str, function) のタプルとする. function は必ず label, pred の順で引数を取るようにする.
    現在のデフォルトでは, N, 正例割合, 予測確率の平均, ROC-AUC, 対数損失, 平均エントロピー, 正規化エントロピー, 期待カリブレーション誤差, 積分カリブレーション指数
    ------
    RETURN: 結果の一覧表を pandas.DataFrame で返す.
    """
    if names is None:
        if len(label_list) == 2:
            names = ('train', 'test')
        elif len(label_list) == 3:
            names = ('train', 'valid', 'test')
        else:
            names = list(range(len(label_list)))
    else:
        pass
    if metrics is None:
        metrics = [
                ('N', lambda y, p: y.shape[0]),
                ('CTR', lambda y, p: y.mean()),
                ('Expecred click', lambda y, p: p.mean()),
                ('AUC', roc_auc_score),
                ('log loss', log_loss),
                ('mean entropy', lambda y, p: entropy(y)),
                ('NE', normalized_entropy),
                ('ECE', expected_calibration_error),
                # ('ICI', integrated_calibration_index)  # 10000 件超えるとめちゃくちゃ遅い
                ]
    result = {name: [metric(y, pi) for y, pi in zip(label_list, pred_list)] for name, metric in metrics}
    return pd.DataFrame(result).assign(model=names)[['model'] + [x[0] for x in metrics]]


def plot_ROC(label_list, pred_list, names=None, **args):
    """
    複数の ROC 曲線をプロットする 
    :param: label_list: 正解ラベルリストの配列. [(y1, y2, ...), (y1, y2, ...)]  のようにして与える,  pred_list に対応させる
    :param: pred_list: 予測確率リストの配列. label_list と同じ長さにすること
    :param: names=None: モデルの名称. None または同じ長さにすること. 指定しない場合,
            ラベルの組が 2~3  ならば ['train', 'valid', 'test'] を与える. 3より多い場合は通し番号にする.
    :param args: sklearn.metrics.roc_curve に与えるパラメータ
    :return: plotnine オブジェクト
    """
    if names is None:
        if len(label_list) == 2:
            names = ('train', 'test')
        elif len(label_list) == 3:
            names = ('train', 'valid', 'test')
        else:
            names = list(range(len(label_list)))
    else:
        pass
    roc = [roc_curve(y, p, **args) for y, p in zip(label_list, pred_list)]
    fpr, tpr = tuple([list(chain.from_iterable(x)) for x in zip(*roc)][0:2])
    models = chain.from_iterable([[name] * l for name, l in zip(names, [len(x) for x, y, _ in roc])])
    d_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'model': models})
    return ggplot(
            d_roc,
            aes(x='fpr', y='tpr', group='model', color='model')
    ) + geom_segment(x=0, y=0, xend=1, yend=1, linetype=':', color='grey'
    ) + geom_line(
    ) + scale_color_discrete(breaks=names
    ) + labs(x='false positive rate', y='true positive rate'
    ) + coord_equal(ratio=1, xlim=[0, 1], ylim=[0, 1]
    ) + theme_classic() + theme(figure_size=(4, 4))


def plot_calibration(label_list, pred_list, names=None, **args):
    """
    カリブレーションカーブを複数描く.
    :param: label_list: 正解ラベルリストの配列. [(y1, y2, ...), (y1, y2, ...)]  のようにして与える,  pred_list に対応させる
    :param: pred_list: 予測確率リストの配列. label_list と同じ長さにすること
    :param: names=None: モデルの名称. None または同じ長さにすること. 指定しない場合, ラベルの組が 2~3  ならば ['train', 'valid', 'test'] を与える. 3より多い場合は通し番号にする.
    :param: args: sklearn.metrics.roc_curve に与えるパラメータ.
        :param: strategy='quantile': 分割方法. 'quantile' または 'uniform'
        :param: n_bins=10: ビン数.
        :param: normalize=False: 予測確率の0-1正規化が必要かどうか
    :return: plotnine オブジェクト
    TODO: 入力データがすごい偏ってるときの表示範囲
    """
    if names is None:
        if len(label_list) == 2:
            names = ('train', 'test')
        elif len(label_list) == 3:
            names = ('train', 'valid', 'test')
        elif len(label_list) == 1:
            names = 'model',
        else:
            names = list(range(len(label_list)))
    else:
        pass
    if args is None:
        args = {'strategy': 'quantile', 'n_bins': 5}
    else:
        args['strategy'] = args['strategy'] if 'strategy' in args.keys() else 'quantile'
        args['n_bins'] = args['n_bins'] if 'n_bins' in args.keys() else 10
    calib = [calibration_curve(y, p, **args) for y, p in zip(label_list, pred_list)]
    frac, pred = tuple([list(chain.from_iterable(x)) for x in zip(*calib)][0:2])
    models = chain.from_iterable([[name] * l for name, l in zip(names, [len(x) for x, y in calib])])
    d_calib = pd.DataFrame({'pred': pred, 'frac': frac, 'model': models})
    return ggplot(
            d_calib,
            aes(x='pred', y='frac', group='model', color='model')
    ) + geom_segment(x=0, y=0, xend=1, yend=1, linetype=':', color='grey'
    ) + geom_line(
    ) + geom_point(
    ) + scale_color_discrete(breaks=names
    ) + labs(x='mean estimated probability', y='fraction of positives'
    ) + coord_equal(ratio=1) + theme_classic() + theme(figure_size=(4, 4))


def plot_pred_hist(label_list, pred_list, names=None, n_bins=10):
    """
    予測確率のヒストグラムを描く
    :param: label_list: 正解ラベルリストの配列. [(y1, y2, ...), (y1, y2, ...)]  のようにして与える,  pred_list に対応させる
    :param: pred_list: 予測確率リストの配列. label_list と同じ長さにすること
    :param: names=None: モデルの名称. None または同じ長さにすること. 指定しない場合, ラベルの組が 2~3  ならば ['train', 'valid', 'test'] を与える. 3より多い場合は通し番号にする.
    :param: n_bins: ヒストグラムのビン数
    :return: plotnine オブジェクト
    TODO: geom_vline の表示方法
    """
    if names is None:
        if len(label_list) == 2:
            names = ('train', 'test')
        elif len(label_list) == 3:
            names = ('train', 'valid', 'test')
        else:
            names = list(range(len(label_list)))
    else:
        pass
    name_order = {k: v for v, k in enumerate(names)}
    name_order_rev = {str(k): v for v, k in name_order.items()}
    d = pd.DataFrame(
            {col: v for col, v in zip(('y', 'prediction'), [list(chain.from_iterable(x)) for x in ([label_list, pred_list])])}
    ).assign(
        model=list(chain.from_iterable([[name] * len(l) for name, l in zip(names, label_list)]))
    ).melt(
        id_vars='model'
    ).assign(
        order=lambda x: x.model.replace(name_order)
    ).sort_values(['order', 'variable'])
    # 補助線としての平均値を引くためのデータ
    d_mean = d.drop(columns='order').groupby(['variable', 'model']).mean(
            ).reset_index().rename(columns={'value': 'mean'})
    d = d.merge(d_mean, on=['variable', 'model'])
    return ggplot(
            d,
            aes(x='value', y='..density..', group='variable', fill='variable')
    ) + geom_histogram(position='identity', alpha=.5, bins=10
    ) + geom_vline(
            aes(xintercept='mean', group='variable', color='variable',
                linetype='variable')
    ) + labs(x='prediction', fill='frequency', linetype='mean', color='mean'
    ) + facet_wrap(
            '~order', scales='free_y', labeller=lambda x: name_order_rev[x]
    ) + theme_classic() + theme(figure_size=(6, 4))


def return_concat_train_test(p_train, p_test, y_train=None, y_test=None, train_label='1_train', test_label='2_test'):
    '''
    p_train, p_test, y_train, y_test を結合して train/test ラベルを付け, 一つの DataFrame にする
    '''
    d = pd.DataFrame({
        'pred': np.concatenate((p_train[:, 0], p_test[:, 0])),
        'data': ['1_train'] * p_train.shape[0] + ['2_test'] * p_test.shape[0]
    })
    if y_train is not None and y_test is not None:
        d['label'] = np.concatenate((y_train, y_test))
    else:
        pass
    return d
