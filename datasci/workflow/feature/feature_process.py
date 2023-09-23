# -*- coding:utf8 -*-
import os
import shelve
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from datasci.workflow.feature.tal_feature_group import FeaturePackage
from datasci.workflow.feature.encoder_map import get_encoder
from datasci.utils.mylog import get_stream_logger
import pandas as pd
import collections

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


class GroupFeatureProcesser(object):
    def __init__(self, process_mode='FeatureGroup', config=None, encoder_map=None, log=None):
        """
            Args
            -------
            model
                model

            process_mode
                Two mode can be selected that `FeatureGroup` & `Pipeline`,
                    `FeatureGroup` is realized by Zhaolihan with columns name after processed,
                    `Pipeline` is based sklearn.pipeline without columns name

            config
                a json string,  e.g.
                    {
                    "process_dag":
                        {
                            "start": "step1",
                            "step1": "step2",
                            "step2": "step3",
                            "step3": "end"
                        },
                    "simple": {
                        "0": {"details": "是否是线索", "is_label": 1, "name": "is_clue", "type": "numberical", "is_use": 1},
                        "1": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "缺勤但看回放的讲次数","is_label": 0, "name": "absent_playback_nums", "type": "numberical", "is_use": 1},
                        "2": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "出勤场均献花金币数","is_label": 0, "name": "avg_praises", "type": "numberical", "is_use": 1},
                        "3": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "出勤场均领取红包次数","is_label": 0, "name": "avg_rec_gold_times", "type": "numberical", "is_use": 1},
                        "4": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "出勤场均领取红包金币数","is_label": 0, "name": "avg_rec_golds", "type": "numberical", "is_use": 1},
                        "5": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "出勤场均发言次数","is_label": 0, "name": "avg_talk_num", "type": "numberical", "is_use": 1},
                        "6": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近30天学习首页次数","is_label": 0, "name": "click_03_92_30day", "type": "numberical", "is_use": 1},
                        "7": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近30天我次数","is_label": 0, "name": "click_05_01_30day", "type": "numberical", "is_use": 1},
                        "8": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近7天我次数","is_label": 0, "name": "click_05_01_7day", "type": "numberical", "is_use": 1},
                        "9": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近30天退费原因弹窗次数","is_label": 0, "name": "click_05_11_30day", "type": "numberical", "is_use": 1},
                        "10": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近30天找客服次数","is_label": 0, "name": "click_05_29_30day", "type": "numberical", "is_use": 1},
                        "11": {"process": {"step1": "mean", "step2": "", "step3": "int"}, "details": "近7天找客服次数","is_label": 0, "name": "click_05_29_7day", "type": "numberical", "is_use": 1},
                    }
            encoder_map
                a json string, e.g.
                        {
                        "ordinal": {
                            "object": "user_intention.model.feature_engineering.encoders.tal_encoder.TalOrdinalEncoder",
                            "params": {"categories" : "auto"}
                        },
                        "most_frequent": {
                            "object": "sklearn.impute.SimpleImputer",
                            "params": { "strategy":"most_frequent"}
                        }

            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("FEATURE", level=log_level) if log is None else log
        if config is None:
            self.log.error("Feature process config must not None!")
            exit(0)
        if encoder_map is None:
            self.log.error("Encoder Map config must not None!")
            exit(0)
        self.simple_feature = config.get('simple', None)
        self.process_dag = config.get('process_dag', None)

        self.encoders = encoder_map
        self.combine_feature = config.get('combine', None)
        self.process_mode = process_mode
        self.columns, self.feature_map, self.label_columns, self.process_config = self._get_config_info()
        self.feature_package = None

    def _get_config_info(self):
        # Sklearn pipeline use process_pipeline config
        process_pipeline = collections.defaultdict(list)

        # Feature Groups use process_groups config
        process_groups = dict()
        feature_map = dict()  # like {"feature1": 1 , "feature2": 2 ...}
        label_columns = list()
        columns = list()  # whitout label columns

        if self.simple_feature is None:
            return None

        for k, v in self.simple_feature.items():
            name = v.get('name')
            is_use = v.get('is_use')
            is_label = v.get('is_label')
            process = v.get('process')

            if is_use == 1:
                feature_map[name] = k
                if is_label != 1:
                    columns.append(name)
                if is_label == 1:
                    label_columns.append(name)
                if process:
                    # get process groups config
                    if self.process_mode == 'FeatureGroup':
                        for encoder_type, encoder in process.items():
                            if encoder_type in process_groups:
                                if encoder in process_groups[encoder_type]:
                                    process_groups[encoder_type][encoder].append(name)
                                else:
                                    process_groups[encoder_type][encoder] = list()
                                    process_groups[encoder_type][encoder].append(name)
                            else:
                                process_groups[encoder_type] = dict()
                                process_groups[encoder_type][encoder] = list()
                                process_groups[encoder_type][encoder].append(name)

                    # get process pipeline config
                    if self.process_mode == 'Pipeline':
                        process_pipeline_list = list()
                        for _, encoder in process.items():
                            if encoder == '' or encoder is None:
                                continue
                            process_pipeline_list.append(encoder)
                        process_pipeline_key = '-'.join(process_pipeline_list)
                        process_pipeline[process_pipeline_key].append(name)
        if self.process_mode == 'FeatureGroup':
            return columns, feature_map, label_columns, process_groups
        if self.process_mode == 'Pipeline':
            return columns, feature_map, label_columns, process_pipeline

    def select_columns(self, data, with_label=False):
        """
            Args
            -------
            data
                input data
            with_label
                whether get label data or not

            Returns
            -------
            data
                when with_label is False
            data, label
                when with_label is True
        """

        if with_label:
            data, label = data.loc[:, self.columns], data.loc[:, self.label_columns]
            return data, label
        else:
            data = data.loc[:, self.columns]
            return data

    def get_feature_processer(self):
        """
            Get the Feature process package from args process mode

            Returns
            -------
            Feature process package
                which is decided by self.process_mode
        """
        result = None
        if self.process_mode == 'Pipeline':
            pipe_list = list()
            if self.process_config == '':
                return None
            for pipeline_key, cols in self.process_config.items():
                pipelines = pipeline_key.split('-')
                encoder_instance_list = list()
                for encoder in pipelines:
                    encoder_instance = get_encoder(encoder_name=encoder, config=self.encoders)
                    encoder_instance_list.append(encoder_instance)
                this_pipeline = (pipeline_key, make_pipeline(*encoder_instance_list), cols)
                pipe_list.append(this_pipeline)
            result = ColumnTransformer(pipe_list)
        if self.process_mode == 'FeatureGroup':
            if self.process_config == '':
                return None
            result = FeaturePackage(init_config=self.process_config, order=self.process_dag, encoder_map=self.encoders,
                                    log=self.log)
        self.feature_package = result
        return result

    def get_feature_package(self):
        return self.feature_package

    def write(self, file):
        """
            Save self
            Args
            -------
            file
                save file path

            Returns
            -------
            None
        """
        with shelve.open(file, 'c') as db:
            db["feature"] = self
            db.close()

    def write_v2(self, file):
        """
            Save self
            Args
            -------
            file
                save file path

            Returns
            -------
            None
        """
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        f.close()

    @staticmethod
    def write_feature_processer(feature_processer, file):
        """
            Save feature processer
            Args
            -------
            feature_processer
                feature processer
            file
                save file path

            Returns
            -------
            None
        """
        with shelve.open(file, 'c') as db:
            db["feature"] = feature_processer
            db.close()

    @staticmethod
    def write_feature_processer_v2(feature_processer, file):
        """
            Save feature processer
            Args
            -------
            feature_processer
                feature processer
            file
                save file path

            Returns
            -------
            None
        """
        with open(file, 'wb') as f:
            pickle.dump(feature_processer, f)
        f.close()

    @staticmethod
    def read_feature_processer(file):
        """
            Read feature processer from file
            Args
            -------
            file
                file path

            Returns
            -------
            GroupFeatureProcesser
        """
        with shelve.open(file, 'r') as db:
            feature_processer = db["feature"]
            db.close()
            return feature_processer

    @staticmethod
    def read_feature_processer_v2(file):
        """
            Read feature processer from file
            Args
            -------
            file
                file path

            Returns
            -------
            GroupFeatureProcesser
        """
        with open(file, 'rb') as f:
            feature_processer = pickle.load(f)
        f.close()
        return feature_processer
