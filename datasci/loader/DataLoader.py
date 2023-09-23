# coding:utf8

import numpy as np
import pandas as pd
import pymysql
import math
import os
from datasci.constant import VALUE_ERROR_VALUE_NOT_NONE_AT_SAME_TIME
import multiprocessing


def load_txt_data(file_name, **kwargs):
    """
      Load data file.
    Args:
      file_name: a str value, file name.
      error: a str value, method of error operating.
      sep: a str value, segment char.

    Returns:
      if sep is not null, return a list[list[]] value, else return a list[str] value.

    """
    encoding = kwargs.get('encoding', 'utf8')
    error = kwargs.get('error', 'strict')
    sep = kwargs.get('sep')
    with open(file_name, 'r', encoding=encoding, errors=error) as fin:
        content = fin.readlines()
        if sep:
            content = [item.replace('\n', '').split(sep) for item in content]
        else:
            content = [item.replace('\n', '') for item in content]
    return content


def load_csv_data(file_name, **kwargs):
    """
      Load CSV data with pandas lib.
      For detail of full parameters, please go to https://www.cnblogs.com/traditional/p/12514914.html.
    Args:
      file_name:      a str value, file name.
      names:          a list value, column's names for data frame.
      header:         a int value, header row num.
      usecols:        a list value, columns you wanna use.
      index_col:      a str value, the column will be indexed.
      dtype:          a str or dict value, data type for each column.
      engine:         a str value, reading engine. default `c`, [`c`,`python`]
    Returns:
      a DataFrame value
    """

    names = kwargs.get('names')
    header = kwargs.get('header', 0)
    usecols = kwargs.get('usecols')
    index_col = kwargs.get('index_col')
    dtype = kwargs.get('dtype', 'object')
    engine = kwargs.get('engine', 'c')
    excel = kwargs.get('excel', False)

    if not excel:
        sep = kwargs.get('set', ',')
        data = pd.read_csv(file_name, engine=engine, sep=sep, names=names, header=header, usecols=usecols,
                           index_col=index_col, dtype=dtype)
    else:
        sheet_name = kwargs.get('sheet_name', 0)
        data = pd.read_excel(file_name, sheet_name=sheet_name, engine=engine, names=names, header=header,
                             usecols=usecols,
                             index_col=index_col, dtype=dtype)

    return data


def load_np_data(file_name, **kwargs):
    """
      Load numpy file.
    Args:
      file_name:    a str value, file name.
      seg:          a str value, segment char.
      dtype:        a numpy dtype value, data type.

    Returns:
      a numpy array
    """
    is_txt = kwargs.get('is_txt')
    sep = kwargs.get('sep')
    dtype = kwargs.get('dtype')

    if is_txt:
        func = np.loadtxt
    else:
        func = np.load

    data = func(file_name, delimiter=sep, dtype=dtype)
    return data


def load_mysql_data(sql, **kwargs):
    """
      Load data from mysql using sql, and return a DataFrame with fixed columns.

    Args:
      sql:          a str value, execute the sql to get data from mysql.
      host:         a str value, host name.
      user:         a str value, mysql user account.
      password:     a str value, mysql password for given user name.
      db:           a str value, mysql database.
      charset:      a str value, database charset.
      columns:      a str or a list value, the columns you wanna use.

    Returns:
      a DataFrame value
    """
    host = kwargs.get('host')
    port = kwargs.get('port')
    user = kwargs.get('user')
    password = kwargs.get('password')
    db = kwargs.get('db')
    charset = kwargs.get('charset')

    assert host is not None and user is not None and password is not None, \
        ValueError('The params {host, user, password} are not allowed be null.')

    con = pymysql.connect(host=host, port=port, user=user, password=password, database=db, charset=charset)
    data = pd.read_sql(sql=sql, con=con)
    columns = kwargs.get('columns')
    if columns is not None:
        data = data[columns]

    return data


def _process_data_subprocessing(idx, manager_dict, data, op_func, kwargs):
    manager_dict[idx] = op_func(data, **kwargs)


def process_data(data, op_func, num_workers=1, **kwargs):
    """
        多进程处理数据
    Args:
        data: 输入数据
        op_func: 处理函数
        num_workers: 进程数
        **kwargs: 其它参数

    Returns:
        处理完成的数据
    """

    def throw_error(e):
        raise e

    func_kwargs = kwargs.get('func_kwargs')
    is_tuple_data = kwargs.get('is_tuple_data')
    if func_kwargs is None:
        func_kwargs = {}
    if num_workers < 2:
        data = op_func(data, **func_kwargs)
    else:
        # 计算每个进程处理的数据量
        if is_tuple_data:
            data_len = len(data[0])
        else:
            data_len = len(data)

        batch_size = math.ceil(data_len / num_workers)
        # 按照单进程数据量对数据集进行切分
        batch_idxs = [list(range(batch_size * idx, min(batch_size * (idx + 1), data_len))) for idx in
                      range(num_workers)]

        # 定义进程池
        pool = multiprocessing.Pool(num_workers)
        # 定义数据共享管理器
        manager = multiprocessing.Manager()
        manager_dict = manager.dict()

        # 向进程池分发任务
        for idx in range(len(batch_idxs)):
            # 分批将数据放入进程
            _ = pool.apply_async(_process_data_subprocessing,
                                 (idx, manager_dict,
                                  data[batch_idxs[idx]] if not is_tuple_data else [item[batch_idxs[idx]] for item in
                                                                                   data],
                                  op_func, func_kwargs), error_callback=throw_error)
        pool.close()
        pool.join()

        data = []
        seg_locale = [0]
        subprocess_result = None
        for idx in range(len(batch_idxs)):
            subprocess_result = manager_dict.get(idx)
            # 判断返回的数据是否为tuple，如果是tuple，则系统默认其返回值是多个
            if isinstance(subprocess_result, tuple):
                if idx == 0:
                    for item in subprocess_result[:-1]:
                        seg_locale.append(item.shape[1] + seg_locale[-1])
                    seg_locale = seg_locale[1:]

                tmp_data = np.concatenate(subprocess_result, axis=1)
            else:
                tmp_data = subprocess_result
            data.append(tmp_data)
        data = np.concatenate(data, axis=0)

        if isinstance(subprocess_result, tuple):
            data = np.split(data, seg_locale, axis=1)
    return data


def data_generator(data, batch_size=128, shuffle=False):
    """
        批数据生成器，用于批量返回数据。在程序中使用
    Args:
        data: ndarray类型，将要分批的数据
        batch_size: 批大小
        shuffle: 是否随机打乱

    Returns:
        一个迭代器
    """
    is_tuple = False
    seg_locale = [0]
    if isinstance(data, tuple):
        is_tuple = True
        for item in data:
            seg_locale.append(item.shape[1] + seg_locale[-1])
        seg_locale = seg_locale[1: -1]
        data = np.concatenate(data, axis=1)

    while True:
        if shuffle:
            np.random.shuffle(data)

        batch_num = len(data) / batch_size
        if batch_num > int(batch_num):
            batch_num += 1
        for idx in range(int(batch_num)):
            batch_data = data[int(idx * batch_size): int(min(len(data), (idx + 1) * batch_size))]
            if is_tuple:
                # 前面的操作会将tuple中所有的数据拼接，这里要进行split分开形成tuple
                yield np.split(batch_data, seg_locale, axis=-1)
            else:
                yield batch_data
