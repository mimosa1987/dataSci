import logging
import os
import json

from datasci.workflow.config.global_config import config_dir

log_file = os.path.join(config_dir, "log_config.json")
lvl = "INFO"
if os.path.exists(log_file):
    with open(log_file) as f:
        l = f.read()
    log_config = json.loads(l)
    lvl = log_config.get("level", None) if log_config.get("level", None) is not None else "INFO"


def _get_loglevel(lvl):
    if lvl == "INFO":
        return logging.INFO
    if lvl == "DEBUG":
        return logging.DEBUG
    if lvl == "CRITICAL":
        return logging.CRITICAL
    if lvl == "ERROR":
        return logging.ERROR
    if lvl == "FATAL":
        return logging.FATAL
    if lvl == "WARN":
        return logging.WARN
    if lvl == "NOTSET":
        return logging.NOTSET


log_level = _get_loglevel(lvl)
