from ..dao.bean.mysql_conf import MySQLConf
from ..constant import VALUE_TYPE_ERROR_TIPS
import pymysql
import pandas as pd
from ..dao import Dao
from sqlalchemy import create_engine
import itertools
import traceback


class MySQLDao(Dao):
    def __init__(self, conf=None, auto_connect=True):
        """

        Args:
          conf: Configuration for the MySQL Connector.
          auto_connect: Whether connecting mysql automatically. default=True.
        """
        super(MySQLDao, self).__init__(conf)
        self.connector = None
        # check the value type of the parameter
        assert isinstance(conf, MySQLConf), ValueError(VALUE_TYPE_ERROR_TIPS)

        self._conf = conf

        if auto_connect:
            self.connect()

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, conf):
        # check the value type of the parameter
        assert isinstance(conf, MySQLConf), ValueError(VALUE_TYPE_ERROR_TIPS)

        self._conf = conf

    def connect(self):
        """
        connect to mysql server

        """
        conf = self.conf
        self.connector = pymysql.connect(host=conf.host, port=conf.port,
                                         user=conf.user, passwd=conf.passwd,
                                         db=conf.db_name)

    def disconnect(self):
        """
        close connector

        """
        if self.is_connected():
            self.connector.close()

    def reconnect(self):
        """
        reconnect to mysql server

        """
        self.disconnect()
        self.connect()

    def is_connected(self):
        """
        check the connector
        Returns:
          bool value
        """
        try:
            self.connector.ping()  # check the connection status
            return True
        except:
            return False

    def load_mysql_data_df(self, sql, wanna_cols=None, return_series=False, verbose=True):
        """
        Execute sql and return fixed result.
        Args:
          sql: a sql with type `str`
          wanna_cols: list of columns which you wanna preserve.
          return_series:
          verbose: bool value, whether print sql.

        Returns:
          data with type `pd.DataFrame`
        """
        if not self.is_connected():
            self.reconnect()
        if verbose:
            print('{}\n{}\n{}'.format('=' * 40, sql, '=' * 40))
        execute_result = pd.read_sql(sql, con=self.connector)
        self.connector.close()

        if wanna_cols is not None:
            if isinstance(wanna_cols, (list, tuple, set)):
                if len(wanna_cols) == 1 and return_series:
                    return execute_result[wanna_cols[0]]
                else:
                    return execute_result[wanna_cols]
            elif isinstance(wanna_cols, str):
                return execute_result[wanna_cols]
            else:
                raise ValueError(VALUE_TYPE_ERROR_TIPS)
        else:
            return execute_result

    def save_data_with_pandas(self, dataframe, table_name, if_exists='append', index=False):
        """
        write DataFrame data to mysql
        Args:
          dataframe: pd.DataFrame data which will be written to mysql
          table_name: str value, mysql table name
          if_exists: str value in (append, fail)
          index: bool value, whether write pd.DataFrame index as a column

        """
        engine = create_engine("mysql+pymysql://{}:{}@{}/{}?charset={}".format(
            self.conf.user, self.conf.passwd, self.conf.host + ":" + str(self.conf.port), self.conf.db_name,
            self.conf.charset))
        connector = engine.connect()
        dataframe.to_sql(name=table_name, con=connector, if_exists=if_exists, index=index)
        connector.close()
        # done.

    def update_data(self, table_name, condition_cols, condition_values, target_cols, target_values, verbose=True):
        """
        update exists data with new value
        Args:
          table_name: str value, table's name
          condition_cols: list<str> value, condition columns
          condition_values: list value, condition values
          target_cols: list<str> value, columns will be updated.
          target_values: list value, new values
          verbose:

        Returns:

        """
        engine = create_engine("mysql+pymysql://{}:{}@{}/{}?charset={}".format(
            self.conf.user, self.conf.passwd, self.conf.host + ":" + str(self.conf.port), self.conf.db_name,
            self.conf.charset))
        connector = engine.connect()
        # generate condition part of string
        condition_str = ' where ' + ' and '.join(['{} = {}'] * len(condition_cols))
        # generate assignment part of string
        set_str = ' set ' + ' and '.join(['{} = {}'] * len(target_cols))
        sql = ''
        for idx in range(len(condition_values)):
            zip_target = zip(target_cols, target_values[idx])
            zip_condition = zip(condition_cols, condition_values[idx])

            sql += ' update {} {} {}; '.format(
                table_name,
                set_str.format(*itertools.chain.from_iterable(
                    [[k, "'%s'" % v if isinstance(v, str) else v] for k, v in
                     zip_target])),
                condition_str.format(*itertools.chain.from_iterable(
                    [[k, "'%s'" % v if isinstance(v, str) else v] for k, v in
                     zip_condition])))
        # execute sql
        if verbose:
            print('{}\n{}\n{}'.format('=' * 40, sql.split(';'), '=' * 40))
        for s in sql.split(';'):
            if len(s.strip()) > 0:
                print('current running:', s)
                connector.execute(s)
        # done.

    def insert_data(self, table_name, cols, values, update_col_when_duplicate=None, duplicate_col_op=None):
        """

        Args:
          table_name:
          cols:
          values:
          update_col_when_duplicate:
          duplicate_col_op:

        Returns:

        """
        engine = create_engine("mysql+pymysql://{}:{}@{}/{}?charset={}".format(
            self.conf.user, self.conf.passwd, self.conf.host + ":" + str(self.conf.port), self.conf.db_name,
            self.conf.charset))
        connector = engine.connect()

        # define placeholder
        values_str = '({}),' * len(values)
        values_str = values_str[:-1]

        # generate values string
        values_str = values_str.format(*[
            ','.join(map(lambda x: str(x), map(lambda x: "'%s'" % x if isinstance(x, str) else x, item))) for item in
            values])

        # generate sql
        # sql = 'replace into {} (id,字段1) values (1,'2'),(2,'3'),...(x,'y');'
        if update_col_when_duplicate is not None:
            if isinstance(update_col_when_duplicate, str):
                update_col_when_duplicate = update_col_when_duplicate.split(',')

        if duplicate_col_op is not None:
            if isinstance(duplicate_col_op, str):
                duplicate_col_op = duplicate_col_op.split(',')

            # duplicate_update_str_list = ['`{}` = values(`{}`) {}'.format(item, item, duplicate_col_op[idx]) for
            #                              idx, item in enumerate(update_col_when_duplicate)]
            duplicate_update_str_list = [
                '`{}` = values(`{}`) {}'.format(item, item, duplicate_col_op[idx].split('|')[1]) \
                    if duplicate_col_op[idx].split('|')[0] != 'origin' \
                    else '`{}` = `{}` {}'.format(item, item, duplicate_col_op[idx].split('|')[1]) for
                idx, item in enumerate(update_col_when_duplicate)]
            duplicate_update_str = ', '.join(duplicate_update_str_list)

            sql = 'insert into {} ({}) values {} on duplicate key update {};'.format(
                table_name, ','.join(cols), values_str, duplicate_update_str)
            # sql = 'insert into {} ({}) values {} on duplicate key update `{}` = values(`{}`);'.format(
            #   table_name, ','.join(cols), values_str, update_col_when_duplicate, update_col_when_duplicate)
        else:
            sql = 'insert into {} ({}) values {} ;'.format(
                table_name, ','.join(cols), values_str)

        connector.execute(sql)

    def execute_sql(self, sql):
        """
        execute sql
        Args:
          sql: str value

        Returns:
          list data
        """
        cursor = self.connector.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        self.connector.commit()
        cursor.close()
        self.connector.close()
        return results

    def execute_sql_with_pandas(self, sql, wanna_cols=None, return_series=False, verbose=True):
        """
        Execute sql and return fixed result.
        Args:
          sql: a sql with type `str`
          wanna_cols: list of columns which you wanna preserve.
          verbose: bool value, whether print sql.

        Returns:
          data with type `pd.DataFrame`
        """
        if not self.is_connected():
            self.reconnect()
        if verbose:
            print('{}\n{}\n{}'.format('=' * 40, sql, '=' * 40))
        execute_result = pd.read_sql(sql, con=self.connector)
        self.connector.close()

        if wanna_cols is not None:
            if isinstance(wanna_cols, (list, tuple, set)):
                if len(wanna_cols) == 1 and return_series:
                    return execute_result[wanna_cols[0]]
                else:
                    return execute_result[wanna_cols]
            elif isinstance(wanna_cols, str):
                return execute_result[wanna_cols]
            else:
                raise ValueError(VALUE_TYPE_ERROR_TIPS)
        else:
            return execute_result

    def insert_data_with_sql(self, sql, data):
        """

        Args:
          sql: 执行sql
          data: 格式举例：
            [('000005', 2, '合肥', 'HZ', '2018-09-19 14:55:21', u'2520.64'),
             ('000006', 2, '北京', 'HZ', '2018-09-19 14:55:21', u'2694.92'),
             ('000007', 2, '上海', 'HZ', '2018-09-19 14:55:21', u'2745.38')]

        Returns:
          the number of data been updated
        """
        cursor = self.connector.cursor()
        num = cursor.executemany(sql, data)
        self.connector.commit()
        cursor.close()
        self.connector.close()
        return num

    def update_data_with_sql(self, sql):
        """
        update mysql data
        Args:
          sql: str value

        Returns:
          the number of data been updated
        """
        cursor = self.connector.cursor()
        num = cursor.execute(sql)
        self.connector.commit()
        cursor.close()
        self.connector.close()
        return num
