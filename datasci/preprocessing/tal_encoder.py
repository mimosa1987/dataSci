from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

"""
自定义类，实现方式：
1、扩展已有的sklearn 已有的encoder，定义函数 column_change ,如 TalOrdinalEncoder 
(原生的OrdinalEncoder 无法将预测集中没有出现的类别进行编码，重写OrdinalEncoder的transform 函数解决问题)
def column_change(self, input_columns, change_feature_map):
    final_cols = input_columns
    return final_cols, change_feature_map 

2、编写Encoder 继承 BaseEstimator, TransformerMixin， 实现fit、transform、fit_transform等函数（函数列表详见父类），
定义 column_change 函数，如 TalWoeEncoder

! 如果不定义 column_change， 系统将使用 column.column_change 函数,此函数也许无法满足你的需求。
"""


class TalOrdinalEncoder(OrdinalEncoder):
    """
        Inherit sklearn.OrdinalEncoder
    """

    def __init__(self, categories='auto', dtype=np.int32):
        self.categories = categories
        self.dtype = dtype

    def transform(self, X):
        """
        Overwrite the function transform of sklearn.OrdinalEncoder,set handle_unknown params to -1
        """
        X_int, _ = self._transform(X, handle_unknown='-1')
        return X_int.astype(self.dtype, copy=False)

    def column_change(self, input_columns, change_feature_map):
        """
        Get change of the encoder which changed the columns
        :param encoder: encoder
        :param input_columns: the columns before chenged
        :param change_feature_map: changed map of features
        :return: the changed cloumns ,and feature map
        """
        final_cols = input_columns
        return final_cols, change_feature_map


class TalOneHotEncoder(OneHotEncoder):

    def column_change(self, input_columns, change_feature_map):
        """
        Get change of the encoder which changed the columns
        :param encoder: encoder
        :param input_columns: the columns before chenged
        :param change_feature_map: changed map of features
        :return: the changed cloumns ,and feature map
        """
        final_cols = list()
        for i in range(len(input_columns)):
            tmp_cols = ['%s_%s' % (input_columns[i], x) for x in self.categories_[i]]
            change_feature_map[input_columns[i]] = tmp_cols
            final_cols.extend(tmp_cols)

        return final_cols, change_feature_map


# Woe Encoder
class TalWoeEncoder(BaseEstimator, TransformerMixin):
    """
    An encoder realized Woe transform , not realize

    """

    def __init__(self):
        print("__init__")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values

    def column_change(self, input_columns, change_feature_map):
        """
        Get change of the encoder which changed the columns
        :param encoder: encoder
        :param input_columns: the columns before chenged
        :param change_feature_map: changed map of features
        :return: the changed cloumns ,and feature map
        """
        final_cols = input_columns
        return final_cols, change_feature_map


# value type encoder
class TalValueTypeEncoder(BaseEstimator, TransformerMixin):
    """
    An encoder realized type transform , not work
    """

    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(self.type)
