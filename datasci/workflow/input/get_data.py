import collections

import pandas as pd
from datasci.utils.reflection import Reflection


def _get_reader_class(reader_cls_str, params):
    idx = reader_cls_str.rfind(".")
    module_path = reader_cls_str[0: idx]
    class_name = reader_cls_str[idx + 1: len(reader_cls_str)]
    if module_path is None or class_name is None:
        return None
    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=params)
    return cls_obj


def get_data(input_args):
    """
        Args
        -------
        input_args
            input config with different data source

        Returns
        -------
        pandas.Dataframe or An iterator
    """
    data_reader = input_args.get('object')
    params = input_args.get('params')
    reader = _get_reader_class(reader_cls_str=data_reader, params=params)
    if isinstance(reader, collections.Iterator):
        return reader
    else:
        return reader.read_data()
