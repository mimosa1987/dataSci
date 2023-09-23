# coding:utf8

from sklearn.externals import joblib


def load_model(filename=None, mmap_mode=None):
    """
        模型加载
      Args:
        filename: 模型读取路径，文件格式支持‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’
        mmap_mode: mmap模式，默认None, 支持‘r+’, ‘r’, ‘w+’, ‘c’

      Returns:
         返回存储数据文件列表
     """
    estimator = joblib.load(filename, mmap_mode)
    return estimator
