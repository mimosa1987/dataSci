import time
import os
from datasci.workflow.config.task_config import get_config
from datasci.workflow.input.get_data import get_data
from datasci.workflow.train.train_package import TrainPackage
from datasci.workflow.feature.feature_process import GroupFeatureProcesser
from datasci.utils.mylog import get_stream_logger
from collections import Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import threading

threadLock = threading.Lock()
threads = []


class MultiTrainThread(threading.Thread):
    def __init__(self, thread_id, processer, model_name, model_config, data=None):
        threading.Thread.__init__(self)
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger('MultiTrainThread: %s' % thread_id, level=log_level)
        self.processer = processer
        self.model_name = model_name
        self.model_config = model_config
        self.data = data

    def run(self):
        threadLock.acquire()
        self.log.info('The thread of %s starting ... ...' % self.model_name)
        threadLock.release()
        self.processer._run(model_name=self.model_name, model_config=self.model_config,
                            data=self.data)
        threadLock.acquire()
        self.log.info('The thread of %s finished' % self.model_name)
        threadLock.release()


class TrainProcesser(object):
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
        self.log = get_stream_logger("TRAIN", level=log_level) if log is None else log
        self.jobs = get_config(config_type="job", config=config)
        self.log.debug("Job config is : %s" % self.jobs)
        self.encoders = get_config(config_type="encoder", config=encoder_map)
        self.log.debug("Encoder config is : %s" % self.jobs)
        self.features = get_config(config_type="feature", config=fconfig)
        self.log.debug("Feature config is : %s" % self.features)
        self.models = get_config(config_type="model", config=model_map)
        self.log.debug("Model config is : %s" % self.models)

        paths = self.jobs.get('paths')
        self.project_path = paths.get('project_path')

        self.data_path = paths.get('data_path')
        self.data_path = os.path.join(self.project_path, self.data_path) if not os.path.isabs(
            self.data_path) else self.data_path

        self.pre_model_path = paths.get('pre_model_path')
        self.pre_model_path = os.path.join(self.data_path, self.pre_model_path) if not os.path.isabs(
            self.pre_model_path) else self.pre_model_path

        self.model_path = paths.get('model_path')
        self.model_path = os.path.join(self.data_path, self.model_path) if not os.path.isabs(
            self.model_path) else self.model_path

        self.feature_process_path = paths.get('feature_package_path')
        self.feature_process_path = os.path.join(self.data_path, self.feature_process_path) if not os.path.isabs(
            self.feature_process_path) else self.feature_process_path

        self.train_data_path = paths.get('train_data_path')
        self.train_data_path = os.path.join(self.data_path, self.train_data_path) if not os.path.isabs(
            self.train_data_path) else self.train_data_path

    def run(self, data=None, multi_process=False):
        models_config = self.jobs.get('models')
        if multi_process:
            i = 0
            for model_name, model_config in models_config.items():
                new_thread = MultiTrainThread(thread_id=i, processer=self, model_name=model_name,
                                              model_config=model_config, data=data)
                new_thread.start()
                threads.append(new_thread)
                i = i + 1
            for t in threads:
                t.join()
        else:
            for model_name, model_config in models_config.items():
                self.log.info('The process of %s starting ... ...' % model_name)
                self._run(model_name=model_name, model_config=model_config, data=data)
                self.log.info('The process of %s finished' % model_name)

    def _run(self, model_name, model_config, data=None):
        need_train = model_config.get('train').get('need_train', False)
        if need_train:
            # Feature args
            feature_process = GroupFeatureProcesser(config=self.features.get(model_name).get('feature_process'),
                                                    encoder_map=self.encoders, log=self.log)
            # data args
            model_input_config = model_config.get('input').get('train_data')
            # evaluate args
            willing = model_config.get('train').get('willing', False)
            save_fpg = model_config.get('train').get('save_fpg', False)
            save_train_data = model_config.get('train').get('save_train_data', False)
            val_prop = model_config.get('train').get('val_prop', 0.2)
            # model args
            model_type = model_config.get('model_type', None)
            model_version = model_config.get('model_version')
            train_package = TrainPackage(
                model_name=model_name,
                model_type=model_type,
                model_version=model_version,
                model_map=self.models,
                log=self.log
            )
            self.train(train_package=train_package, feature_process=feature_process,
                       feature_process_path=self.feature_process_path, input_config=model_input_config, data=data,
                       val_prop=val_prop, save_gfp=save_fpg, willing=willing, save_path=self.model_path,
                       save_train_data=save_train_data)

    def train(self, train_package, feature_process, feature_process_path=None, input_config=None, data=None,
              val_prop=0.2, save_gfp=False, willing=None, save_path=None, save_train_data=False):
        """
             Predict data and save result, get data from config
             Args
             -------
             train_package
                An instance of TrainPackage

             feature_process
                An instance of FeaturePrcoesser

             feature_process_path
                The path save feature processer

             input_config
                A dict like
                    {
                        "data_type": "mysql",
                        "file": {
                            "path": "/Users/zhaolihan/Documents/TAL/git/xes1v1_data_science/user_intention/data_source/datafile/train/data.csv"
                        },
                        "mysql": {
                            "sql": "/Users/zhaolihan/Documents/TAL/git/xes1v1_data_science/user_intention/data_source/sql/train_data.sql",
                            "args": {
                                "batch_size": 100000,
                                "max_iter": -1,
                                "offset": 0,
                                "batch_output": true
                        }
                    }
             data
                 data
             val_prop
                 the proportion of valuation data
             save_gfp
                 Whether save GroupFeatureProcesser

            willing

            save_path
                The path where model saved
            Returns
            -------
            result
                None
        """
        run_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        tdata = get_data(input_config) if data is None else data
        fpr = feature_process.get_feature_processer()

        model_file_name = "%s_%s_%s.model" % (train_package.model_name, train_package.model_type, run_time)
        model_file_path = os.path.join(save_path, model_file_name)

        file_name = "%s_%s_%s.pickle.fper" % (train_package.model_name, train_package.model_type, run_time)
        feature_process_file = os.path.join(feature_process_path, file_name)

        if isinstance(tdata, Iterable) and not isinstance(tdata, pd.DataFrame):
            X_train_datas = list()
            X_val_datas = list()
            X_test_datas = list()
            y_train_datas = list()
            y_val_datas = list()
            y_test_datas = list()
            index = 0
            for data in tdata:
                if save_train_data:
                    tmp_train_data_file = os.path.join(self.train_data_path,
                                                       "%s_%s_%s.data" % (train_package.model_name, run_time, index))
                    data.to_csv(tmp_train_data_file)
                    index += 1

                data[data.isnull()] = np.NaN
                _data, _label = feature_process.select_columns(data=data, with_label=True)
                X_train, X_val, y_train, y_val = train_test_split(_data, _label, test_size=val_prop,
                                                                  train_size=(1.0 - val_prop), random_state=1024)
                X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5,
                                                                train_size=0.5, random_state=1024)
                X_train_datas.append(X_train)
                X_val_datas.append(X_val)
                X_test_datas.append(X_test)
                y_train_datas.append(y_train)
                y_val_datas.append(y_val)
                y_test_datas.append(y_test)

            self.log.info('%s getting feature processer ... ' % train_package.model_name)
            all_train_data = pd.concat(X_train_datas, axis=0)
            all_train_data.reset_index(drop=True, inplace=True)
            fpr.fit(all_train_data)
            del all_train_data
            if save_gfp:
                GroupFeatureProcesser.write_feature_processer_v2(feature_process, feature_process_file)
                self.log.info('Model %s Saved group feature process in dir : %s' % (
                    train_package.model_name, feature_process_file))
            pre_model = None
            for i in range(len(X_train_datas)):
                self.log.info('Training data batch %s ... ' % i)
                X_train = X_train_datas[i]
                y_train = y_train_datas[i]
                X_val = X_val_datas[i]
                y_val = y_val_datas[i]
                X_test = X_test_datas[i]
                y_test = y_test_datas[i]

                X_train = fpr.transform(X_train)
                X_val = fpr.transform(X_val)
                X_test = fpr.transform(X_test)
                model = train_package.train(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                                            pre_model=pre_model)
                pre_model = model
                # Evalating ... ...
                self.log.info('Evalating model %s  ... ...' % train_package.model_name)
                train_ret = train_package.evaluate(X_val=X_train, y_val=y_train)
                self.log.info('Train data size %s, %s Evaluated result : %s ' % (
                    X_train.shape[0], train_package.model_name, train_ret))
                eval_ret = train_package.evaluate(X_val=X_val, y_val=y_val)
                self.log.info('Valuate data size %s, %s Evaluated result : %s ' % (
                    X_val.shape[0], train_package.model_name, eval_ret))
                test_ret = train_package.evaluate(X_val=X_test, y_val=y_test, willing=willing,
                                                  save_path=model_file_path)
                self.log.info('Test data size %s, %s Evaluated result : %s ' % (
                    X_test.shape[0], train_package.model_name, test_ret))
        else:
            if save_train_data:
                tmp_train_data_file = os.path.join(self.train_data_path,
                                                   "%s_%s.data" % (train_package.model_name, run_time))
                tdata.to_csv(tmp_train_data_file)

            tdata[tdata.isnull()] = np.NaN
            _data, _label = feature_process.select_columns(data=tdata, with_label=True)
            X_train, X_val, y_train, y_val = train_test_split(_data, _label, test_size=val_prop,
                                                              train_size=(1.0 - val_prop), random_state=1024)
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5,
                                                            train_size=0.5, random_state=1024)
            self.log.info('%s getting feature processer ... ' % train_package.model_name)
            X_train = fpr.fit_transform(X_train)

            if save_gfp:
                GroupFeatureProcesser.write_feature_processer_v2(feature_process, feature_process_file)
                self.log.info('Model %s Saved group feature process in dir : %s' % (
                    train_package.model_name, feature_process_file))

            # Training ... ...
            X_val = fpr.transform(X_val)
            X_test = fpr.transform(X_test)
            train_package.train(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

            # Evalating ... ...
            self.log.info('Evalating model %s  ... ...' % train_package.model_name)
            train_ret = train_package.evaluate(X_val=X_train, y_val=y_train, willing=willing)
            self.log.info('Train data size %s, %s Evaluated result : %s ' % (
                X_train.shape[0], train_package.model_name, train_ret))
            eval_ret = train_package.evaluate(X_val=X_val, y_val=y_val, willing=willing)
            self.log.info('Valuate data size %s, %s Evaluated result : %s ' % (
                X_val.shape[0], train_package.model_name, eval_ret))
            test_ret = train_package.evaluate(X_val=X_test, y_val=y_test, willing=willing, is_test=False,
                                              save_path=model_file_path)
            self.log.info(
                'Test data size %s, %s Evaluated result : %s ' % (X_test.shape[0], train_package.model_name, test_ret))
