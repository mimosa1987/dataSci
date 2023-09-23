import functools

from datasci.workflow.model.model_map import get_outclass_function, get_object
from sklearn import metrics


def _class_func_params(m_type, func_name, model, config=None):
    """
    :param m_type:  model type
    :param func_name: function name
    :param model: model instance
    :param config: config
    :return:
    """
    type_function_map = None
    out_class_function_map = get_outclass_function('model', config=config)
    if out_class_function_map is not None:
        type_function_map = out_class_function_map.get(m_type, None)
    if type_function_map is not None and func_name in type_function_map:
        func, params = out_class_function_map.get(m_type).get(func_name)
        return None, func, params
    else:
        if model is None:
            return None, None, None
        obj, func_dict = get_object(object_name=m_type, model=model, config=config)
        if func_dict is not None and type(obj) == type(model) and func_name in func_dict:
            func, params = func_dict.get(func_name).get('func', None), func_dict.get(func_name).get('params', None)
            return model, func, params
        else:
            return None, None, None


def load_model_func(m_type, model=None, func_params=None, config=None):
    """
        Load model from model file

        Args
        -------
        m_type
            Model type like xgb | xgb_r which defined yourself
        model
            An instance of Model which type is model type

        config
            Hot config


        Returns
        -------
        model
    """
    _, func, params = _class_func_params(m_type=m_type, func_name='load_model', model=model, config=config)
    all_params = func_params
    if params is not None:
        all_params = dict(func_params, **params)
    ret_model = func(**all_params)
    return ret_model


def save_model_func(m_type, model=None, func_params=None, config=None):
    """
        Save model to model file

        Args
        -------
        m_type
            Model type like xgb | xgb_r which defined yourself
        model
            An instance of Model which type is model type

        config
            Hot config


        Returns
        -------
        None
    """
    _, func, params = _class_func_params(m_type=m_type, func_name='dump_model', model=model, config=config)
    all_params = func_params
    if params is not None:
        all_params = dict(func_params, **params)
    func(model, **all_params)


def train_func(m_type, model=None, func_params=None, config=None):
    """
       Train the model

        Args
        -------
        m_type
            Model type like xgb | xgb_r which defined yourself
        model
            An instance of Model which type is model type

        config
            Hot config

        Returns
        -------
        model
    """
    _, func, params = _class_func_params(m_type=m_type, func_name='train', model=model, config=config)
    all_params = func_params
    if params is not None:
        all_params = dict(func_params, **params)
    model = func(**all_params)
    return model


def predict_func(m_type, model=None, func_params=None, config=None):
    """
       Predict data with model

        Args
        -------
        m_type
            Model type like xgb | xgb_r which defined yourself
        model
            An instance of Model which type is model type

        config
            Hot config

        Returns
        -------
        Output of model

    """
    _, func, params = _class_func_params(m_type=m_type, func_name='predict', model=model, config=config)
    all_params = func_params
    if params is not None:
        all_params = dict(func_params, **params)
    return func(**all_params)


def predict_proba_func(m_type, model=None, func_params=None, config=None):
    """
       Predict data with model

        Args
        -------
        m_type
            Model type like xgb | xgb_r which defined yourself
        model
            An instance of Model which type is model type

        config
            Hot config

        Returns
        -------
        Output of model

    """
    _, func, params = _class_func_params(m_type=m_type, func_name='predict_proba', model=model, config=config)
    all_params = func_params
    if params is not None:
        all_params = dict(func_params, **params)
    return func(**all_params)


def _get_eval_indicator_func(indicator_name='auc'):
    """
    Get eval function

    """
    if indicator_name == 'auc':
        return metrics.roc_auc_score
    if indicator_name == 'f1':
        return metrics.f1_score
    if indicator_name == 'recall':
        return metrics.recall_score
    if indicator_name == 'precision':
        return metrics.precision_score


# def model_eval(model, X, y, willing={'auc': 0.65}):
#     pass

def eval_func(m_type, X, y, model=None, willing=None, config=None, is_test=True, save_path=None):
    """
       Eval model

    """
    metrics_dict = {}
    access = True
    kw = {"m_type": m_type, "model": model, "config": config}
    get_function = functools.partial(_class_func_params, **kw)
    _, func, _ = get_function(func_name='predict')
    _, func_proba, _ = get_function(func_name='predict_proba')
    true_rate = 0.0
    for metric, willing_value in willing.items():
        metric_eval_func = _get_eval_indicator_func(metric)
        if metric == 'auc':
            try:
                pred = func_proba(X)[:, 1]
            except IndexError:
                pred = func_proba(X)
        else:
            pred = func(X)
            true_rate = (pred.sum() / pred.shape[0])
        value = metric_eval_func(y, pred)
        metrics_dict[metric] = value
        if is_test or value < willing_value:
            access = False
    metrics_dict['true_rate'] = true_rate
    if access:
        _, save_func, params = get_function(func_name='dump_model')
        params = {"model_file": save_path}
        save_func(**params)
    return metrics_dict
