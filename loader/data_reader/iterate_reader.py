# -*- coding:utf8 -*-

from datasci.utils.mylog import get_stream_logger
from sqlalchemy import create_engine
import pandas as pd
import time
import os


class MySQLIterateDataReader(object):
    """
        Mysql data reader (a Iterator)
    """

    def __init__(self,
                 section='Mysql-data_bank',
                 batch_size=1000000,
                 offset=0,
                 max_iter=10,
                 sql='',
                 func=None,
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
        self.batch_size = batch_size
        self.offset = offset
        self.max_iter = max_iter
        self.func = func
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("MYSQL ITERATE DATA READER", level=log_level) if log is None else log

    def __iter__(self):
        self.log.info('Iterate start ... ...')
        return self

    def __next__(self):
        if self.max_iter != 0:
            # 获取MySQL数据
            sql = '%s limit %s offset %s ' % (self.sql, self.batch_size, self.offset)
            self.log.debug('Read data from sql script: %s ' % sql)
            self.log.info('Read data from offset : %s ' % self.offset)

            try:
                df = pd.read_sql(sql, self.engine)
            except Exception as e:
                self.log.error('SQL failed! Reason : %s ' % e)
            # 判断最后一次迭代，并处理数据后退出
            cur_batch_size = len(df)
            self.log.info('Batch size : %s' % cur_batch_size)

            if cur_batch_size == 0:
                self.log.info('Iterate over ... ...')
                raise StopIteration
            self.log.info(' The iterate : %s --> %s finished, finished time : %s' % (
                self.offset, self.offset + cur_batch_size,
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

            self.offset = self.offset + cur_batch_size if cur_batch_size < self.batch_size else self.offset + self.batch_size
            self.max_iter = self.max_iter - 1 if self.max_iter > 0 else self.max_iter

            if self.func is None:
                return df
            else:
                retlist = df.values.tolist()
                ret = list()
                for data in retlist:
                    ret.append(self.func(data))
                return ret

        self.log.info('Iterate over ... ...')
        raise StopIteration
