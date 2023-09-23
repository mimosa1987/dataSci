from datasci.utils.mylog import get_stream_logger
from datasci.workflow.config.task_config import get_config
from datasci.workflow.output.save_data import save_data

class SaveProcesser(object):
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
        self.log = get_stream_logger("SAVE", level=log_level) if log is None else log
        self.jobs = get_config(config_type="job", config=config)
        self.log.debug("Job config is : %s" % self.jobs)
        self.output = self.jobs.get('output')

    def run(self, data, output_config=None, extend_columns=None, pagesize=10000):
        """

        :param data: pd.Dataframe
        :param output_config:  like

            {
                "output": {
                    "object": "datasci.dumper.data_writer.batch_writer.MySQLDataWriter",
                    "params": {
                        "table": "ads_user_intention_join_model_predict_result",
                        "section": "Doris-xes1v1_db"
                    }
                }
            }
        :param data_tag: like 'join' in the json string of output_config
        :param extend_columns: extend columns in a dict
        :param pagesize: how much lines to write in one step
        :return:
        """

        output_conf = self.output if output_config is None else output_config
        size = data.shape[0]
        batch_num = size // pagesize + 1
        for i in range(batch_num):
            begin = i * pagesize
            end = begin + pagesize if begin + pagesize < size else size
            batch_data = data.iloc[begin: end]
            save_d = batch_data.copy()
            if extend_columns is not None:
                for col_name, col_value in extend_columns.items():
                    save_d[col_name] = col_value
            self.log.info('Save data which batch number is %s ' % i)
            save_data(data=save_d, output_args=output_conf)
