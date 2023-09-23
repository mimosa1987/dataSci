
from datasci.workflow.model.model_func import train_func, eval_func

import pandas as pd
from datasci.workflow.model.model_map import get_object
import warnings
from datasci.utils.mylog import get_stream_logger

log = get_stream_logger("TrainPackage")
warnings.filterwarnings("ignore")
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


class TrainPackage(object):
    def __init__(self, model_name, model_type,  model_version, model_map=None, log=None):
        """
            A package of predict

            Args
            -------
            model_name
                model name

            model_type
                Model type

            model_version
                The model version
            model_map
                model map
            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("TRAIN", level=log_level) if log is None else log
        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        self.model_map = model_map
        self.model, _ = get_object(object_name=model_type, config=model_map)
        self.id = "tp_%s" % self.model_name

    def evaluate(self,X_val, y_val, willing=None, is_test=True,  save_path=None):
        # Eval & Save
        result = eval_func(m_type=self.model_type, model=self.model, X=X_val, y=y_val, willing=willing,config=self.model_map,
                           is_test=is_test, save_path=save_path)
        return result

    def train(self, X_train=None, X_val=None, y_train=None, y_val=None, pre_model=None):
        train_params = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "pre_model": pre_model
        }
        self.model = train_func(m_type=self.model_type, model=self.model, func_params=train_params,
                                config=self.model_map)
