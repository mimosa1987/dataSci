# coding:utf8
from functools import partial
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from pandas import DataFrame
import math


class WOE_IV(object):
    """
    WOE IV编码器
    """

    def __init__(self, df: DataFrame, cols_to_woe: [str], label_column: str, good_label: str):
        """

        Args:
            df: spark dataframe对象
            cols_to_woe: list值，列名称列表
            label_column: label列的名称
            good_label: 正例的值
        """
        self.df = df
        self.cols_to_woe = cols_to_woe
        self.label_column = label_column
        self.good_label = good_label
        self.fit_data = {}

    def fit(self):
        """
        计算WOE和IV

        """
        for col_to_woe in self.cols_to_woe:
            total_good = self.compute_total_amount_of_good()
            total_bad = self.compute_total_amount_of_bad()

            woe_df = self.df.select(col_to_woe)
            categories = woe_df.distinct().collect()
            for category_row in categories:
                category = category_row[col_to_woe]
                good_amount = self.compute_good_amount(col_to_woe, category)
                bad_amount = self.compute_bad_amount(col_to_woe, category)

                good_amount = good_amount if good_amount != 0 else 0.5
                bad_amount = bad_amount if bad_amount != 0 else 0.5

                good_dist = good_amount / total_good
                bad_dist = bad_amount / total_bad

                self.build_fit_data(col_to_woe, category, good_dist, bad_dist)

    def transform(self, df: DataFrame):
        """
        对数据进行WOE编码
        Args:
            df: spark dataframe对象

        Returns:
            编码后的spark dataframe对象
        """

        def _encode_woe(col_to_woe_):
            return F.coalesce(
                *[F.when(F.col(col_to_woe_) == category, F.lit(woe_iv['woe']))
                  for category, woe_iv in self.fit_data[col_to_woe_].items()]
            )

        for col_to_woe, woe_info in self.fit_data.items():
            df = df.withColumn(col_to_woe + '_woe', _encode_woe(col_to_woe))
        return df

    def compute_total_amount_of_good(self):
        return self.df.select(self.label_column).filter(F.col(self.label_column) == self.good_label).count()

    def compute_total_amount_of_bad(self):
        return self.df.select(self.label_column).filter(F.col(self.label_column) != self.good_label).count()

    def compute_good_amount(self, col_to_woe: str, category: str):
        return self.df.select(col_to_woe, self.label_column) \
            .filter(
            (F.col(col_to_woe) == category) & (F.col(self.label_column) == self.good_label)
        ).count()

    def compute_bad_amount(self, col_to_woe: str, category: str):
        return self.df.select(col_to_woe, self.label_column) \
            .filter(
            (F.col(col_to_woe) == category) & (F.col(self.label_column) != self.good_label)
        ).count()

    def build_fit_data(self, col_to_woe, category, good_dist, bad_dist):
        woe_info = {
            category: {
                'woe': math.log(good_dist / bad_dist),
                'iv': (good_dist - bad_dist) * math.log(good_dist / bad_dist)
            }
        }

        if col_to_woe not in self.fit_data:
            self.fit_data[col_to_woe] = woe_info
        else:
            self.fit_data[col_to_woe].update(woe_info)

    def compute_iv(self):
        iv_dict = {}

        for woe_col, categories in self.fit_data.items():
            iv_dict[woe_col] = 0
            for category, woe_iv in categories.items():
                iv_dict[woe_col] += woe_iv['iv']
        return iv_dict


def remove_cols(df, cols):
    """
    删除列
    Args:
        df: spark dataframe对象
        cols: list值，列名称列表

    Returns:
        删除指定列后的spark dataframe对象
    """
    for col in cols:
        df = df.drop(col)
    return df


def woe_encode(df, cols, label_col, mapping_broadcast, good_label='1', remove_old_cols=False):
    """
    woe编码
    Args:
        df: spark dataframe对象
        cols: list值，列名称列表
        label_col: label的列名称
        mapping_broadcast: 添加了广播的变量，编码映射器
        good_label: 正例的值
        remove_old_cols：list值，要移除的列名称

    Returns:
        编码后的spark dataframe对象
    """
    mapping = mapping_broadcast.value
    woe_model = WOE_IV(df, cols, label_col, good_label)

    woe_model.fit_data = mapping
    df = woe_model.transform(df)

    if remove_old_cols:
        df = remove_cols(df, cols)

    return df


def onehot_encode(df, cols, mapping_broadcast, remove_old_cols=False):
    """
    onehot编码
    Args:
        df: spark dataframe对象
        cols: list值，列名称列表
        mapping_broadcast: 添加了广播的变量，编码映射器
        remove_old_cols: list值，要移除的列名称

    Returns:
        编码后的spark dataframe对象
    """

    def onehot_func(x, y):
        return 1 if x == y else 0

    mapping = mapping_broadcast.value

    for col in cols:  # 遍历每列
        onehot_seq = mapping[col]

        for val in onehot_seq:  # 遍历某列的每个值，包含空值，空值这里被替换为'nan'
            onehot_parfunc = partial(onehot_func, y=val)
            # 用Spark的UDF封装我们自定义的偏函数
            onehot_parfunc_udf = F.udf(onehot_parfunc, IntegerType())
            # 使用计算结果生成新列
            df = df.withColumn(
                col + '_' + val, onehot_parfunc_udf(df[col]))

    if remove_old_cols:
        df = remove_cols(df, cols)

    return df
