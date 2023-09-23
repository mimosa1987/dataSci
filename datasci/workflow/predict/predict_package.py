import os
from datasci.workflow.model.model_func import load_model_func, predict_func, predict_proba_func
from datasci.utils.mylog import get_file_logger, get_stream_logger


class PredictPackage(object):
    def __init__(self, model_name, model_file, model_type, model_version, model_map=None, log=None):
        """
        A package of predict

        Args
        -------
        model_name
            model name

        model_file
            Model file

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
        self.log = get_stream_logger("PREDICT", level=log_level) if log is None else log
        self.model_name = model_name
        self.model_file = model_file
        self.model_type = model_type
        self.id = "pp_%s" % self.model_name
        self.model_version = model_version
        self.model_map = model_map

    def predict(self, data):
        """
        Predict data and save result

        Args
        -------
        data
            predict data

        Returns
        -------
        result
            result and save
        """

        model = self._get_model()
        predict_params = {
            "data": data
        }

        result = predict_func(m_type=self.model_type, model=model, func_params=predict_params, config=self.model_map)
        return result

    def predict_proba(self, data):
        """
        Predict data and save result

        Args
        -------
        data
            predict data

        Returns
        -------
        result
            result and save
        """

        model = self._get_model()
        predict_params = {
            "data": data
        }

        result = predict_proba_func(m_type=self.model_type, model=model, func_params=predict_params, config=self.model_map)
        return result

    def _get_model(self):
        if not os.path.exists(self.model_file):
            self.log.error("Model file %s is not exists ! " % self.model_file)
            exit(-1)
        load_model_func_params = {"filename": self.model_file}
        model = load_model_func(m_type=self.model_type, model=None, func_params=load_model_func_params)
        return model
