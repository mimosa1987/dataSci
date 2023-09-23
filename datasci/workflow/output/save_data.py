from datasci.utils.reflection import Reflection


def _get_writer_class(writer_cls_str, params):
    idx = writer_cls_str.rfind(".")
    module_path = writer_cls_str[0: idx]
    class_name = writer_cls_str[idx + 1: len(writer_cls_str)]
    if module_path is None or class_name is None:
        return None
    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=params)
    return cls_obj


def save_data(output_args, data):
    """
        Args
        -------
        output_args
            output config with different data source

    """
    data_writer = output_args.get('object')
    params = output_args.get('params')
    writer = _get_writer_class(writer_cls_str=data_writer, params=params)
    return writer.save_data(data=data)
