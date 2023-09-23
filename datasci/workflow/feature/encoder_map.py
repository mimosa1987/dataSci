from datasci.utils.reflection import Reflection

def get_encoder(encoder_name, config=None):
    """
        Select encoder from encoder name
        Args
        -------
        encoder_name
            encoder name

        config
            encoder map like the content in 'conf/encoder_config.json'

        config_file
            encoder map file e.g. 'conf/encoder_config.json'

        Returns
        -------
        Encoder
    """

    return _get_encoder_from_encoder_dict(
        _get_encoder_dict(encoder_name=encoder_name, config=config))


def _get_encoder_dict(encoder_name, config=None):
    return config.get(encoder_name, None)


def _get_encoder_from_encoder_dict(encoder_dict):
    """
        Select encoder use reflection
        Args
        -------
        encoder_dict
            a dict like
            {
            "encoder" :"user_intention.model.feature_engineering.encoders.user_defined_encoder.SelfOrdinalEncoder",
            "params" : "categories='auto', dtype=np.int32"
            }

        Returns
        -------
        Encoder
    """

    if encoder_dict is None or encoder_dict == "":
        return None
    encoder = encoder_dict.get("object", None)
    idx = encoder.rfind(".")
    module_path = encoder[0: idx]
    class_name = encoder[idx + 1: len(encoder)]
    if module_path is None:
        return None
    if class_name is None:
        return None
    params = encoder_dict.get("params", None)

    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=params)
    return cls_obj
