from datasci.utils.mylog import get_stream_logger
from datasci.workflow.config.task_config import get_config
import pandas as pd

class JoinProcesser(object):
    def __init__(self, config=None, log=None):
        """
            A Join instance

            Args
            -------
            config
                Job config dict , which like the content of "conf/job_config.json"
            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("JOIN", level=log_level) if log is None else log
        self.jobs = get_config(config_type="job", config=config)
        self.log.debug("Job config is : %s" % self.jobs)
        self.join = self.jobs.get('join')
        self.join_key = self.join.get('join_key')

    def run(self, data, join_key=None):
        """
        :param data_dict: data dict
        :param join_key:  join  key
        :return: join data
        """
        result = pd.DataFrame()
        join_key = self.join_key if join_key is None else join_key
        if isinstance(data, dict):
            value_data = data.values()
        else:
            value_data = data
        for item in value_data:
            result = pd.merge(result, item, on=join_key) if join_key in result.columns else item
        return result
