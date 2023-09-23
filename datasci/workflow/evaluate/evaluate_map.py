from datasci.utils.reflection import Reflection

def get_evaluator(evaluate_config):
    """
        Select evaluate use reflection
        Args
        -------
        evaluate_dict
            a dict like
            {
            "object" :"user_intention.model.feature_engineering.encoders.user_defined_encoder.SelfOrdinalEncoder",
            "params" : "categories='auto', dtype=np.int32"
            }

        Returns
        -------
        Encoder
    """

    if evaluate_config is None or evaluate_config == "":
        return None
    evaluate_func = evaluate_config.get("object", None)
    idx = evaluate_func.rfind(".")
    module_path = evaluate_func[0: idx]
    class_name = evaluate_func[idx + 1: len(evaluate_func)]
    if module_path is None:
        return None
    if class_name is None:
        return None
    params = evaluate_config.get("params", None)

    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=params)
    return cls_obj
