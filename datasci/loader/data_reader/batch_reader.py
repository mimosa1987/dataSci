# -*- coding:utf8 -*-
from datasci.utils.mylog import get_stream_logger
from sqlalchemy import create_engine
import pandas as pd
import os


class MySQLDataReader(object):
    """
        Mysql data reader (a Iterator)
    """

    def __init__(self,
                 section='Mysql-data_bank',
                 sql='',
                 host=None,
                 port=None,
                 user=None,
                 password=None,
                 db=None,
                 encoding=None,
                 log=None
                 ):
        section = str(section)
        from datasci.utils.read_config import get_global_config
        host = get_global_config(section, 'host') if host is None else host
        port = int(get_global_config(section, 'port')) if port is None else port
        user = get_global_config(section, 'user') if user is None else user
        password = get_global_config(section, 'password') if password is None else password
        db = get_global_config(section, 'db') if db is None else db
        encoding = get_global_config(section, 'charset') if encoding is None else encoding
        connect_str = r'mysql+pymysql://{username}:{password}@{host}:{port}/{databases}'.format(
            username=user,
            password=password,
            host=host,
            port=port,
            databases=db)
        self.engine = create_engine(connect_str, encoding=encoding)
        if os.path.exists(sql):
            with open(sql) as f:
                self.sql = f.read()
        else:
            self.sql = sql
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("MYSQL BATCH DATA READER", level=log_level) if log is None else log

    def read_data(self):
        # 获取MySQL数据
        ret = None
        self.log.debug('Reading data from SQL Script : %s ' % self.sql)
        self.log.info('Reading data from SQL ... ...')
        try:
            ret = pd.read_sql(self.sql, self.engine)
        except Exception as e:
            self.log.error('SQL failed! Reason : %s ' % e)
        return ret


class CSVFileDataReader(object):
    """
        csv file  data reader (a Iterator)
    """

    def __init__(self, file='', log=None):
        self.file = file
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("FILE BATCH DATA READER", level=log_level) if log is None else log

    def read_data(self):
        # 获取MySQL数据
        ret = None
        self.log.debug('Reading data from file : %s ' % self.file)
        try:
            ret = pd.read_csv(self.file)
        except Exception as e:
            self.log.error('Read file failed! Reason : %s ' % e)
        return ret
