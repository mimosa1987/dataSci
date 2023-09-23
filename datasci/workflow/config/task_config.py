import json
from datasci.utils.path_check import check_path

def _get_file_config(config_type, config):
    if config is None:
        from datasci.workflow.config.global_config import global_config
        config = global_config.get(config_type)
        check_path(config, is_make=True)
    with open(config) as f:
        conf = f.read()
        return json.loads(conf)


def get_config(config_type, config):
    if config is not None and isinstance(config, dict):
        return config
    elif config is not None and _is_json(config):
        return json.loads(config)
    else:
        return _get_file_config(config_type=config_type, config=config)


def _is_json(string):
    try:
        json.loads(string)
    except ValueError:
        return False
    return True
