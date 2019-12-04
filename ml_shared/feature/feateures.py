#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: s_katagiri
"""

import numpy as np
import scipy as sp
import pandas as pd
from warnings import warn


def hivemall2csr_mat(td_job, feature_list={}, td_label_index=0, td_feature_index=1, extra_cols=[]):
    """
    hivemall の入力データ形式のテーブルを一般的な配列に変換する
    :param job: ラベルと特徴量列のあるテーブルを呼ぶクエリ. 特徴量列は array() である前提.
    :param feature_list: dict. 特徴量名と位置インデックスの対で指定する. 1以上の長さにだと,
                               その名前の特徴量だけを取り出す. デフォルトでは空
    :param td_label_index: integer. ラベル列の位置. デフォルト0
    :param td_feature_index: integer. 特徴量列の位置. デフォルト1
    :param extra_cols: array-like. それ以外の取得したい列. index または str.
    :return: tuple of nunpy.ndarray(label),
                      scipy.sparse.csr_matrix(feature matrix),
                      dict(feature name: feature type),
                      pandas.DataFrame(extra columns)
    TODO: ラベル列の指定だけ分けているのはダサい?
    """
    if td_job.num_records < 1:
        ValueError('`td_job` result table has no records!')
    label = []
    csr_data = []
    csr_row_index = []
    csr_col_index = []
    extra_data = []
    if feature_list:
        is_train_data = False
    else:
        is_train_data = True
    # array 以外の列処理について
    if not extra_cols:
        pass
    else:
        td_job_columns = np.array([name for name, typ in td_job.result_schema])
        if all([type(x) is int for x in extra_cols]):
            if max(extra_cols) > len(td_job_columns):
                warn('`extra_data_index` > TD table columns index. automatically indices are reduced.')
                extra_data_index = [x for x in extra_cols if x <= len(td_job_columns)]
            extra_data_columns = td_job_columns[extra_data_index]
        elif all([type(x) is str for x in extra_cols]):
            extra_data_columns = extra_cols
            if all([x in td_job_columns for x in extra_data_columns]):
                pass
            else:
                warn('some columns which are not included in the table are automatically removed:' + ', '.join(
                    [x for x in extra_data_columns if x not in td_job_columns]) + '. ')
                extra_data_columns = [x for x in extra_cols if x in td_job_columns]
            extra_data_index = [idx for idx, name in enumerate(td_job_columns) if name in extra_data_columns]
        else:
            TypeError('all of the elements in `extra_data_index` must be either str or integer')
    # TODO: pytd への対応
    for row_num, td_row_values in enumerate(td_job.result()):
        label.append(td_row_values[td_label_index])
        if td_row_values[td_feature_index]:
            for col in td_row_values[td_feature_index]:
                key_value = col.split(':')
                if not is_train_data and key_value[0] not in feature_list:
                    pass
                else:
                    index = feature_list.setdefault(key_value[0], len(feature_list))
                    csr_data.append(float(key_value[1]) if len(key_value) > 1 else 1.0)
                    csr_row_index.append(row_num)
                    csr_col_index.append(index)
        extra_data.append([x for i, x in enumerate(td_row_values) if i in extra_data_index])
    extra_data = pd.DataFrame(extra_data, columns=extra_data_columns)
    return (np.array(label),
            sp.sparse.csr_matrix((csr_data, (csr_row_index, csr_col_index)),
                                 shape=(csr_row_index[-1] + 1, len(feature_list)),  # shape 指定したほうが少し速くなる?
                                 dtype=float),
            feature_list,
            extra_data
            )
