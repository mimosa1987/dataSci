import json
from datasci.utils.reflection import Reflection

func_conf_default = {

    "load_model": {
        "object": "tal_load_model",
        "params": {},
        "in_class": True
    },
    "dump_model": {
        "object": "tal_save_model",
        "params": {},
        "in_class": True
    },
    "train": {
        "object": "tal_train",
        "params": {},
        "in_class": True
    },
    "predict": {
        "object": "tal_predict",
        "params": {},
        "in_class": True
    },
    "predict_proba": {
        "object": "tal_predict_proba",
        "params": {},
        "in_class": True
    }
}


def get_simple_object(object_name, config=None):
    """
        Select encoder from encoder name
        Args
        -------
        encoder_name
            encoder name

        config
            encoder map like the content in 'conf/encoder_config.json'

        Returns
        -------
        Encoder
    """

    return _get_object(object_dict=config.get(object_name))


def _get_object(object_dict):
    """
        Select encoder use reflection
        Args
        -------
        object_dict
            a dict like
            {
            "object" :"user_intention.model.feature_engineering.encoders.user_defined_encoder.SelfOrdinalEncoder",
            "params" : "categories='auto', dtype=np.int32"
            }

        Returns
        -------
        Object
    """

    if object_dict is None or object_dict == "":
        return None
    object_class_path = object_dict.get("object", None)
    idx = object_class_path.rfind(".")
    module_path = object_class_path[0: idx]
    class_name = object_class_path[idx + 1: len(object_class_path)]
    if module_path is None or class_name is None:
        return None
    params = object_dict.get("params", None)
    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=params)
    return cls_obj


def get_object(object_name, model=None, config=None):
    """
        Select function from function name
        Args
        -------
        object_name
            object name ,like xgb |onehot, that is the alias of actual name in object_config.json

        object_type
            like model|encoder ,that is the catagory of object

        function_name
            function name logically, its mapped to object_config.json to  finds actual function

        config
            object map like the content in 'conf/model_config.json'

        config_file
            object map file e.g. 'conf/model_config.json'

        Returns
        -------
        object,function_dict
    """
    func_dict = dict()
    conf = config.get(object_name, None)
    obj = _get_object(object_dict=conf) if model is None else model

    func_conf = conf.get('function', func_conf_default)

    if func_conf is not None:
        for fname, func_details in func_conf.items():
            tmp = dict()
            _, func, params = _get_inclass_function(obj, func_details)
            if func is not None:
                tmp['func'] = func
                tmp['params'] = params
                func_dict[fname] = tmp
    else:
        return obj, None
    return obj, func_dict


def _get_inclass_function(obj, function_dict):
    func_obj = function_dict.get("object", None)
    func_params = function_dict.get("params", None)
    in_class = function_dict.get("in_class", None)
    if in_class:
        return obj, Reflection.reflect_func(obj, func_obj), func_params
    else:
        return None, None, None


def get_outclass_function(object_type, config=None, config_file=None):
    out_func = dict()
    _config = config_file
    if _config is None:
        from datasci.workflow.config.global_config import global_config
        _config = global_config.get(object_type, None)
    with open(_config) as f:
        conf = f.read()
        file_config = json.loads(conf)
    obj_dict = file_config if config is None else config
    for obj_type, obj_config in obj_dict.items():

        func_config = obj_config.get("function", func_conf_default)
        if func_config is not None:
            tmp = dict()
            for func_name, func_details in func_config.items():
                if not func_details.get("in_class"):
                    func_obj = func_details.get("object", None)
                    func_params = func_details.get("params", None)
                    idx = func_obj.rfind(".")
                    module_path = func_obj[0: idx]
                    fname = func_obj[idx + 1: len(func_obj)]
                    func, params = Reflection.reflect_obj_func(module_path, func_name=fname), func_params
                    tmp[func_name] = (func, params)
        else:
            return None
        out_func[obj_type] = tmp
    return out_func
