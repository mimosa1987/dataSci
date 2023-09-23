# -*- coding:utf-8 -*-
import os
from collections import Iterable
import numpy as np
import threading

from datasci.workflow.feature.feature_process import GroupFeatureProcesser
from datasci.workflow.predict.predict_package import PredictPackage
from datasci.workflow.input.get_data import get_data
import pandas as pd
from datasci.utils.mylog import get_stream_logger
from datasci.workflow.config.task_config import get_config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

threadLock = threading.Lock()
threads = []


class MultiPredictThread(threading.Thread):
    def __init__(self, thread_id, processer, model_name, model_config, result_dict, data=None):
        threading.Thread.__init__(self)
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger('MultiPredictThread: %s' % thread_id, level=log_level)
        self.processer = processer
        self.model_name = model_name
        self.model_config = model_config
        self.data = data
        self.result_dict = result_dict

    def run(self):
        threadLock.acquire()
        self.log.info('The thread of %s starting ... ...' % self.model_name)
        threadLock.release()
        self.processer._run(model_name=self.model_name, model_config=self.model_config,
                            result_dict=self.result_dict, data=self.data)
        threadLock.acquire()
        self.log.info('The thread of %s finished' % self.model_name)
        threadLock.release()


class PredictProcesser(object):

    def __init__(self, config=None, model_map=None, log=None):
        """
            A packaging of predict process

            Args
            -------
            config
                Job config dict , which like the content of "conf/job_config.json"
            config_file
                Job config file path ,e.g. "conf/job_config.json"
            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("PREDICT", level=log_level) if log is None else log
        self.jobs = get_config(config_type="job", config=config)
        self.log.debug("Job config is : %s" % self.jobs)
        self.models = get_config(config_type="model", config=model_map)
        self.log.debug("Model config is : %s" % self.models)

        paths = self.jobs.get('paths')
        self.project_path = paths.get('project_path')

        self.data_path = paths.get('data_path')
        self.data_path = os.path.join(self.project_path, self.data_path) if not os.path.isabs(
            self.data_path) else self.data_path

        self.model_path = paths.get('model_path')
        self.model_path = os.path.join(self.data_path, self.model_path) if not os.path.isabs(
            self.model_path) else self.model_path

        self.feature_process_path = paths.get('feature_package_path')
        self.feature_process_path = os.path.join(self.data_path, self.feature_process_path) if not os.path.isabs(
            self.feature_process_path) else self.feature_process_path

        self.predict_data_path = paths.get('predict_data_path')
        self.predict_data_path = os.path.join(self.data_path, self.predict_data_path) if not os.path.isabs(
            self.predict_data_path) else self.predict_data_path

    def run(self, data=None, multi_process=False):
        models_config = self.jobs.get('models')
        result_dict = dict()
        if multi_process:
            i = 0
            for model_name, model_config in models_config.items():
                new_thread = MultiPredictThread(thread_id=i, processer=self, model_name=model_name,
                                                model_config=model_config, result_dict=result_dict, data=data)
                new_thread.start()
                threads.append(new_thread)
                i = i + 1
            for t in threads:
                t.join()
            return result_dict
        else:
            for model_name, model_config in models_config.items():
                self.log.info('The process of %s starting ... ...' % model_name)
                self._run(model_name=model_name, model_config=model_config, result_dict=result_dict, data=data)
                self.log.info('The process of %s finished' % model_name)
        if len(result_dict) == 1:
            keys = list(result_dict.keys())
            return result_dict.get(keys[0])
        else:
            return result_dict

    def _run(self, model_name, model_config, result_dict, data=None):
        is_online = model_config.get('predict').get('is_online', False)
        is_proba = model_config.get('predict').get('is_proba', False)
        retain_cols = model_config.get('predict').get('retain_cols', None)
        if is_online:
            model_type = model_config.get('model_type', None)

            feature_process_file = model_config.get('predict').get('feature_package_file', None)
            full_feature_process_file = os.path.join(self.feature_process_path,
                                                     feature_process_file) if not os.path.isabs(
                feature_process_file) else feature_process_file
            if not os.path.exists(full_feature_process_file):
                self.log.error("Feature group process file %s is not exists ! " % full_feature_process_file)
                exit(-1)
            feature_process = GroupFeatureProcesser.read_feature_processer_v2(full_feature_process_file)

            model_file = model_config.get('predict').get('model_file', None)
            full_model_file = os.path.join(self.model_path, model_file) if not os.path.isabs(model_file) else model_file

            input_config = model_config.get('input').get('predict_data')

            model_version = model_config.get('model_version')
            predict_package = PredictPackage(
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                model_file=full_model_file,
                model_map=self.models,
                log=self.log
            )
            ret = self.predict(predict_package=predict_package, feature_process=feature_process,
                               input_config=input_config, data=data, is_proba=is_proba, retain_cols=retain_cols)
            result_dict[model_name] = ret
        return result_dict

    def predict(self, predict_package, feature_process, data=None, input_config=None, is_proba=False, retain_cols=None):
        """
        Predict data and save result, get data from config
        Args
            -------
            predict_package
                An instance of PredictPackage

            feature_process
                An instance of FeaturePrcoesser

            data
                predict data

            input_config
                input config

            is_proba
                proba output

        Returns
        -------
        result
            result and save
        """
        if data is not None:
            tdata = data
        else:
            tdata = get_data(input_config)
        result = pd.DataFrame()
        if isinstance(tdata, Iterable) and not isinstance(tdata, pd.DataFrame):
            for data in tdata:
                data[data.isnull()] = np.NaN
                data.drop_duplicates(inplace=True)
                if retain_cols is None:
                    retain_data = data
                else:
                    retain_data = data[retain_cols]
                self.log.info('%s feature engineering starting ... ...' % predict_package.model_name)
                select_data = feature_process.select_columns(data=data)
                feature_package = feature_process.get_feature_package()
                predict_data = feature_package.transform(select_data)
                if is_proba:
                    ret_data = predict_package.predict_proba(predict_data)
                else:
                    ret_data = predict_package.predict(predict_data)
                try:
                    class_num = ret_data.shape[1]
                except IndexError:
                    class_num = 1
                result_col = ['%s_%s' % (predict_package.model_name, i) for i in range(class_num)]
                ret = pd.concat((retain_data, pd.DataFrame(ret_data, columns=result_col)), axis=1)
                if result.empty:
                    result = ret
                else:
                    result = pd.concat((result, ret), axis=0)
        else:
            tdata[tdata.isnull()] = np.NaN
            tdata.drop_duplicates(inplace=True)

            if retain_cols is None:
                retain_data = tdata
            else:
                retain_data = tdata[retain_cols]

            self.log.info('%s feature engineering starting ... ...' % predict_package.model_name)
            select_data = feature_process.select_columns(data=tdata)
            feature_package = feature_process.get_feature_package()
            predict_data = feature_package.transform(select_data)

            if is_proba:
                ret_data = predict_package.predict_proba(predict_data)
            else:
                ret_data = predict_package.predict(predict_data)
            try:
                class_num = ret_data.shape[1]
            except IndexError:
                class_num = 1
            result_col = ['%s_%s' % (predict_package.model_name, i) for i in range(class_num)]
            result = pd.concat((retain_data, pd.DataFrame(ret_data, columns=result_col)), axis=1)
        result.reset_index(drop=True, inplace=True)
        return result
