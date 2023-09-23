
import pandas as pd
from datasci.workflow.feature.encoder_map import get_encoder
from datasci.workflow.feature.column import column_change
import numpy as np
import shelve
import collections
import copy
from scipy.sparse import csr_matrix, csc_matrix


class FeatureEncoder(object):
    """
    Feature encoder the encoder which sklearn defined and self defined.
    Add columns in its propety
    """

    def __init__(self, encoder_name, columns, encoder_map=None):

        """
        Parameters
        ----------
        encoder_name
            encoder name
        columns
            colums

        Returns
        -------
        None
        """

        self.encoder_name = encoder_name
        self.encoder = get_encoder(encoder_name, config=encoder_map)
        self.input_columns = columns
        self.columns = columns
        self.params = dict()
        self.change_feature_map = collections.defaultdict(list)
        self.is_fit = False
        if encoder_name == '' or encoder_name is None:
            self.id = 'null'
        else:
            self.id = "fe_%s" % self.encoder_name

    def fit(self, X, y=None):
        if self.encoder is None:
            self.is_fit = True
            return self
        self.encoder.fit(X)
        if hasattr(self.encoder, 'column_change'):
            self.columns, _ = self.encoder.column_change(self.input_columns, self.change_feature_map)
        else:
            self.columns, _ = column_change(self.encoder, self.input_columns, self.change_feature_map)
        self.is_fit = True
        return self

    def transform(self, X):
        if self.is_fit == False:
            exit(-1)
        if self.encoder is None:
            return X.values
        data = self.encoder.transform(X)
        if isinstance(data, csr_matrix) or isinstance(data, csc_matrix):
            data = data.toarray()
        return data

    def fit_transform(self, X, y=None):
        if self.encoder is None:
            self.is_fit = True
            return X.values
        data = self.encoder.fit_transform(X)
        if isinstance(data, csr_matrix) or isinstance(data, csc_matrix) :
            data = data.toarray()
        if hasattr(self.encoder, 'column_change'):
            self.columns, _ = self.encoder.column_change(self.input_columns, self.change_feature_map)
        else:
            self.columns, _ = column_change(self.encoder, self.input_columns, self.change_feature_map)
        self.is_fit = True
        return data


class FeatureEncoderGroup(object):
    """
    Feature package group realized a group of  FeatureEncoder's fit/ transform / fit_transform
    """

    def __init__(self, encoder_group_config, encoder_map=None):
        """
            Parameters
            ----------
            encoder_group_config
                encoder group config

            Returns
            -------
            None
        """
        self.encoder_group_config = encoder_group_config
        self.encoder_map = encoder_map
        self.columns = list()
        self.params = dict()
        self.id_str = None
        self.id = None
        self.change_feature_map = collections.defaultdict(list)
        self.is_fit = False
        self.encoder_dict, self.encoder_list = self._parse_config()

    def _parse_config(self):
        encoder_dict = dict()
        encoder_list = list()
        for encoder_name, cols in self.encoder_group_config.items():
            fe = FeatureEncoder(encoder_name=encoder_name, columns=cols, encoder_map=self.encoder_map )
            encoder_dict[encoder_name] = fe
            id_name = encoder_name
            if encoder_name == '' or encoder_name is None:
                id_name = 'null'
            if self.id_str is None:
                self.id_str = "_%s" % id_name
            else:
                self.id_str = "%s_%s" % (self.id_str, id_name)
            encoder_list.append(fe)
        self.id = "feg_%s" % self.id_str
        return encoder_dict, encoder_list

    def fit(self, X, y=None):
        for fe in self.encoder_list:
            fe.fit(X=X[fe.input_columns], y=y)
            self.columns.extend(fe.columns)
            self.change_feature_map = dict(self.change_feature_map, **fe.change_feature_map)
            self.params = dict(self.params, **fe.params)
            self.is_fit = fe.is_fit
        return self

    def transform(self, X):
        if self.is_fit == False:
            exit(-1)
        data_list = list()
        for fe in self.encoder_list:
            data = fe.transform(X=X[fe.input_columns])
            data_list.append(data)
        return np.hstack(tuple(data_list))

    def fit_transform(self, X, y=None):
        data_list = list()
        for fe in self.encoder_list:
            data = fe.fit_transform(X=X[fe.input_columns], y=y)
            data_list.append(data)
            self.columns.extend(fe.columns)
            self.change_feature_map = dict(self.change_feature_map, **fe.change_feature_map)
            self.params = dict(self.params, **fe.params)
            self.is_fit = fe.is_fit
        return np.hstack(tuple(data_list))


class FeaturePackage(object):
    """
    Feature package is a process of the whole feature process.
    Its includes serval Feature group in it and realize the fit/ fit_transform / transform func in it
    """

    def __init__(self, init_config, order, encoder_map=None, log=None):
        """
            Parameters
            ----------
            init_config
                the init config of FeaturePackage
            order
                the DAG config order

            Returns
            -------
            None
        """
        from datasci.utils.mylog import get_stream_logger
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("FEATURE", level=log_level) if log is None else log
        self.is_fit = False
        self.config = init_config
        self.order = order
        self.columns = list()
        self.params = dict()
        self.feg_list = list()
        self.encoder_map = encoder_map
        self.id = None

    def _prase_dag(self):
        order_list = list()
        n = self.order['start']
        while n is not None and n != 'end':
            order_list.append(n)
            n = self.order[n]
        return order_list

    def _update_conf(self, feature_map, encoder_group_config):
        if feature_map == None:
            ret_config = copy.deepcopy(encoder_group_config)
            return ret_config
        if len(feature_map) == 0:
            ret_config = copy.deepcopy(encoder_group_config)
            return ret_config
        for encoder_name, cols in encoder_group_config.items():
            new_cols = list()
            for col in cols:
                if col in feature_map:
                    new_cols.extend(feature_map[col])
                else:
                    new_cols.append(col)
            encoder_group_config[encoder_name] = new_cols
        ret_config = copy.deepcopy(encoder_group_config)
        return ret_config

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        if self.is_fit == False:
            self.log.error('Please fit your data first!')
            exit(-1)
        data = X
        for feg in self.feg_list:
            ret = feg.transform(X=data)
            if isinstance(ret, pd.DataFrame):
                data = ret
            else:
                data = pd.DataFrame(ret, columns=feg.columns)
        return data

    def fit_transform(self, X, y=None):
        updated_conf = None
        data = X
        id_str = None
        for encode_type in self._prase_dag():
            encoder_group_config = self.config.get(encode_type, None)
            if encoder_group_config is None:
                self.log.error("Encoder Group Config is NULL!")
                exit(-1)
            if updated_conf == None:
                updated_conf = self._update_conf(None, encoder_group_config)
            else:
                updated_conf = self._update_conf(feg.change_feature_map, encoder_group_config)
            self.config[encode_type] = updated_conf
            feg = FeatureEncoderGroup(encoder_group_config=updated_conf, encoder_map=self.encoder_map)
            if id_str is None:
                id_str = feg.id_str
            else:
                id_str = "%s%s" % (id_str, feg.id_str)
            self.id = 'fp%s' % id_str
            fit_data = feg.fit_transform(X=data, y=y)
            if isinstance(fit_data, pd.DataFrame):
                data = fit_data
            else:
                data = pd.DataFrame(fit_data, columns=feg.columns)
            self.feg_list.append(feg)
            self.columns = feg.columns
            self.params = feg.params
            self.is_fit = feg.is_fit
        return data

    def dump(self, dist_file):
        if self.is_fit:
            db = shelve.open(dist_file)
            db['feature_package'] = self
            db.close()
        else:
            self.log.error("Feature package is unfiting!")

    @staticmethod
    def dump_feature_package(feature_package, dist_file):
        """
            Save feature package
            Args
            -------
            feature_package
                feature package
            dist_file
                save file path

            Returns
            -------
            None
        """
        if feature_package.is_fit:
            db = shelve.open(dist_file)
            db['feature_package'] = feature_package
            db.close()
        else:
            from datasci.utils.mylog import get_stream_logger
            from datasci.workflow.config.log_config import log_level
            log = get_stream_logger("FEATURE", level=log_level)
            log.error("Feature package is unfiting!")

    @staticmethod
    def load_feature_package(source_file):
        """
            Read feature package  from file
            Args
            -------
            source_file
                file path

            Returns
            -------
            FeaturePackage
        """
        db = shelve.open(source_file)
        feature_package = db['feature_package']
        db.close()
        return feature_package

