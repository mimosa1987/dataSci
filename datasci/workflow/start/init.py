from datasci.workflow.config.global_config import *
from datasci.workflow.config.task_config import get_config
from datasci.utils.path_check import check_path
from datasci.utils.mylog import get_stream_logger
import os


class InitProcesser(object):
    def __init__(self, config=None, fconfig=None, encoder_map=None, model_map=None, log=None):
        """
            A packaging of train

            Args
            -------
            config
                Job config dict , which like the content of "conf/job_config.json"
            config_file
                Job config file path ,.eg "conf/job_config.json"
            fconfig
                Feature config, which like the content of "conf/feature_config.json"
            fconfig_file
                Feature config,e.g. "conf/feature_config.json"

            encoder_map
                encoder map , which like the content of "conf/encoder_config.json"
            encoder_map_file
                encoder map,e.g. "conf/encoder_config.json"

            model_map
                model map, which like the content of "conf/model_config.json"
            model_map_file
                model map,,e.g. "conf/model_config.json"

            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("INIT", level=log_level) if log is None else log


    def run(self):
        # check paths
        check_path(PROJECT_PATH, is_make=True)
        self.log.info("Project path is : %s" % PROJECT_PATH)
        check_path(config_dir, is_make=True)
        self.log.info("Config data path is : %s" % config_dir)
        check_path(log_dir, is_make=True)
        self.log.info("Log data path is : %s" % log_dir)
        check_path(data_dir, is_make=True)
