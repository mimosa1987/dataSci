import os
import pandas as pd
from datasci.dumper.data_writer.batch_writer import MySQLDataWriter
from datasci.dao.mysql_dao import MySQLDao
from datasci.dao.bean.mysql_conf import MySQLConf
from datasci.loader.data_reader.batch_reader import MySQLDataReader
from datasci.utils.mysql_utils import MysqlUtils
from datasci.workflow.node.base import BaseNode


class MysqlReadNode(BaseNode):

    def run(self):
        sql = self.run_params.get('sql', None) if self.run_params is not None else None
        section = self.run_params.get('section', None) if self.run_params is not None else None
        dict_out = self.run_params.get('dict_out', None) if self.run_params is not None else False
        if os.path.exists(sql):
            with open(sql) as f:
                sql = f.read()
        reader = MySQLDataReader(section=section, sql=sql)
        ret = reader.read_data()
        if dict_out:
            ret_d = dict()
            ret_d[self.node_name] = ret
            self.output_data = ret_d
        else:
            self.output_data = ret
        self.is_finished = True
        return self.output_data


class MysqlExecNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        section = self.run_params.get('section', None) \
            if self.run_params is not None else "Mysql-data_bank"
        sql = self.run_params.get('sql', None) \
            if self.run_params is not None else None

        if os.path.exists(sql):
            with open(sql) as f:
                sql = f.read(sql)

        mysql_utils = MysqlUtils(section)
        result = [tuple(x) for x in self.input_data.values]
        try:
            if sql is not None:
                mysql_utils.get_executemany_sql(sql, result)
        except Exception as e:
            print(e)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class MysqlUpdateNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        section = self.run_params.get('section', None) if self.run_params is not None else None
        from datasci.utils.read_config import get_global_config
        host = get_global_config(section, 'host')
        port = int(get_global_config(section, 'port'))
        user = get_global_config(section, 'user')
        password = get_global_config(section, 'password')
        db = get_global_config(section, 'db')
        charset = get_global_config(section, 'charset')
        mysql_conf = MySQLConf(host=host, port=port, user=user, passwd=password, db_name=db, charset=charset)
        mysql_dao = MySQLDao(mysql_conf)
        condition_cols = self.run_params.get('condition_cols',
                                             None) if self.run_params is not None else None
        condition_values = self.run_params.get('condition_values',
                                               None) if self.run_params is not None else self.input_data.loc[:,
                                                                                         condition_cols].values.tolist()
        table_name = self.run_params.get('table_name', None) if self.run_params is not None else None
        target_cols = self.run_params.get('target_cols',
                                          None) if self.run_params is not None else None
        self.input_data = self.input_merge()
        update_data = self.input_data.loc[:, target_cols].values.tolist()
        mysql_dao.update_data(table_name=table_name, condition_cols=condition_cols,
                              condition_values=condition_values,
                              target_cols=target_cols, target_values=update_data)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class MysqlInsertNode(BaseNode):

    def run(self):
        section = self.run_params.get('section', None) if self.run_params is not None else None
        from datasci.utils.read_config import get_global_config
        host = get_global_config(section, 'host')
        port = int(get_global_config(section, 'port'))
        user = get_global_config(section, 'user')
        password = get_global_config(section, 'password')
        db = get_global_config(section, 'db')
        charset = get_global_config(section, 'charset')
        mysql_conf = MySQLConf(host=host, port=port, user=user, passwd=password, db_name=db, charset=charset)
        mysql_dao = MySQLDao(mysql_conf)
        target_cols = self.run_params.get('target_cols',
                                          None) if self.run_params is not None else self.input_data.columns.tolist()
        table_name = self.run_params.get('table_name', None) if self.run_params is not None else None
        update_col_when_duplicate = self.run_params.get('update_col_when_duplicate',
                                                        None) if self.run_params is not None else None
        duplicate_col_op = self.run_params.get('duplicate_col_op',
                                               None) if self.run_params is not None else None
        self.input_data = self.input_merge()
        data = self.input_data.loc[:, target_cols].values.tolist()
        mysql_dao.insert_data(table_name=table_name, cols=target_cols, values=data,
                              update_col_when_duplicate=update_col_when_duplicate, duplicate_col_op=duplicate_col_op)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class DataStackNode(BaseNode):
    def run(self):
        axis = self.run_params.get('axis', None) if self.run_params is not None else 0
        self.output_data = self.input_merge(axis=axis)
        self.is_finished = True
        return self.output_data


class DataMergeNode(BaseNode):
    def run(self):
        result = None
        on = self.run_params.get('on', None) if self.run_params is not None else None
        how = self.run_params.get('how', None) if self.run_params is not None else None
        left_on = self.run_params.get('left_on', None) if self.run_params is not None else None
        right_on = self.run_params.get('right_on', None) if self.run_params is not None else None
        for data in self.input_data:
            result = pd.merge(result, data, how=how, on=on, left_on=left_on,
                              right_on=right_on) if result is not None else data
        self.output_data = result
        self.is_finished = True
        return self.output_data


class SelectDataFromDict(BaseNode):
    def run(self):
        self.input_data = self.input_merge()
        if not isinstance(self.input_data, dict):
            self.output_data = self.input_data
        else:
            keys = list(self.input_data.keys())
            tag = self.run_params.get('tag', None) \
                if self.run_params is not None else keys[0]
            self.output_data = self.input_data.get(tag, None)
        self.is_finished = True
        return self.output_data


class SelectColumnsNode(BaseNode):
    def run(self):
        self.input_data = self.input_merge()
        columns = self.run_params.get('columns', None) \
            if self.run_params is not None else self.input_data.columns.tolist()
        if columns is not None:
            self.output_data = self.input_data[columns]
        self.is_finished = True
        return self.output_data


class SaveMySQLWithDataFrameNode(BaseNode):
    def run(self):
        section = self.run_params.get('section', None) if self.run_params is not None else None
        table_name = self.run_params.get('table_name', None) if self.run_params is not None else None
        is_flush = self.run_params.get('is_flush', None) if self.run_params is not None else  False
        self.input_data = self.input_merge()
        save_class = MySQLDataWriter(section=section, table=table_name, is_flush=is_flush)
        save_class.save_data(self.input_data)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data
