# -*- coding:utf8 -*-

from datasci.utils.mylog import get_stream_logger
from sqlalchemy import create_engine


class MySQLDataWriter(object):
    """
        Mysql data writer
    """

    def __init__(self,
                 section='Mysql-data_bank',
                 table='',
                 is_flush= False,
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
        self.table = table
        if is_flush:
            self.engine.execute("TRUNCATE table %s;" % table)
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("MYSQL BATCH DATA WRITER", level=log_level) if log is None else log

    def save_data(self, data):
        try:
            data.to_sql(self.table, self.engine, if_exists='append', index=False)
        except Exception as e:
            self.log.error('SQL failed! Reason : %s ' % e)


class CSVFileDataWriter(object):
    """
        csv file data writer
    """

    def __init__(self, file='', log=None):
        self.file = file
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("FILE BATCH DATA WRITER", level=log_level) if log is None else log

    def save_data(self, data):
        try:
            data.to_csv(self.file)
        except Exception as e:
            self.log.error('Saving to file  failed! Reason : %s ' % e)
