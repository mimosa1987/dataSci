# -*- coding:utf-8 -*-
import os
from collections import Iterable
import numpy as np
import threading

from datasci.workflow.evaluate.evaluate_map import get_evaluator
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


class MultiEvaluateThread(threading.Thread):
    def __init__(self, thread_id, processer, model_name, model_config, result_dict, data=None):
        threading.Thread.__init__(self)
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger('MultiEvaluateThread: %s' % thread_id, level=log_level)
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


class EvaluateProcesser(object):

    def __init__(self, config=None, model_map=None, log=None):
        """
            A packaging of evaluate process

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
        self.log = get_stream_logger("EVALUATE", level=log_level) if log is None else log
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

        self.evaluate_data_path = paths.get('evaluate_data_path')
        self.evaluate_data_path = os.path.join(self.data_path, self.evaluate_data_path) if not os.path.isabs(
            self.evaluate_data_path) else self.evaluate_data_path

    def run(self, data=None, multi_process=False):
        models_config = self.jobs.get('models')
        result_dict = dict()
        if multi_process:
            i = 0
            for model_name, model_config in models_config.items():
                new_thread = MultiEvaluateThread(thread_id=i, processer=self, model_name=model_name,
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
        is_online = model_config.get('evaluate').get('is_online', False)
        label_column = model_config.get('evaluate').get('label_column', False)
        eval_functions = model_config.get('evaluate').get('eval_functions', False)
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

            input_config = model_config.get('input').get('evaluate_data')

            model_version = model_config.get('model_version')
            predict_package = PredictPackage(
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                model_file=full_model_file,
                model_map=self.models,
                log=self.log
            )
            ret = self.evaluate(predict_package=predict_package, feature_process=feature_process,
                                input_config=input_config, label_column=label_column, data=data,
                                eval_functions=eval_functions)
            result_dict[model_name] = ret
        return result_dict

    def evaluate(self, predict_package, feature_process, data=None, label_column=None, input_config=None,
                 eval_functions=None):
        """
        evaluate data and save result, get data from config
        Args
            -------
            evaluate_package
                An instance of EvaluatePackage

            feature_process
                An instance of FeaturePrcoesser

            data
                evaluate data

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
                ret = self._evaluate(data=data, label_column=label_column, predict_package=predict_package,
                                     feature_process=feature_process)
                if result.empty:
                    result = ret
                else:
                    result = pd.concat((result, ret), axis=0)
        else:
            result = self._evaluate(data=tdata, label_column=label_column, predict_package=predict_package,
                                    feature_process=feature_process)
        result.reset_index(drop=True, inplace=True)
        eval_results = dict()
        for eval_function in eval_functions:
            eval_func = get_evaluator(eval_function)
            eval_result = eval_func.evaluate(result)
            eval_results.update(eval_result)
        return eval_results

    def _evaluate(self, data, label_column, predict_package, feature_process):
        data[data.isnull()] = np.NaN
        data.drop_duplicates(inplace=True)
        target_data = data[label_column]
        self.log.info('%s feature engineering starting ... ...' % predict_package.model_name)
        select_data = feature_process.select_columns(data=data)
        feature_package = feature_process.get_feature_package()
        predict_data = feature_package.transform(select_data)
        proba_predict_data = predict_package.predict_proba(predict_data)
        cls_predict_data = predict_package.predict(predict_data)
        try:
            class_num = proba_predict_data.shape[1]
        except IndexError:
            class_num = 1
        result_col = ['proba_%s' % i for i in range(class_num)]
        ret_arr = np.hstack((target_data.values.reshape(-1, 1), proba_predict_data, cls_predict_data.reshape(-1, 1)))
        result_cols = ['label'] + result_col + ['cls']
        ret = pd.DataFrame(ret_arr, columns= result_cols)
        return ret
