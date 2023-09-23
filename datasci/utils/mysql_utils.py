# -*- coding:utf-8 -*-
from datasci.utils.read_config import get_global_config
import pymysql
import pandas as pd


class MysqlUtils(object):

    def __init__(self, section):
        self.section = str(section)
        self.host = get_global_config(self.section, 'host')
        self.port = int(get_global_config(self.section, 'port'))
        self.user = get_global_config(self.section, 'user')
        self.password = get_global_config(self.section, 'password')
        self.db = get_global_config(self.section, 'db')
        self.charset = get_global_config(self.section, 'charset')
        self.connect = r'mysql+pymysql://{username}:{password}@{host}:{port}/{databases}'.format(
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            databases=self.db)

    def get_result_sql(self, sql):
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        self.cursor = self.connect.cursor()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.connect.commit()
        self.cursor.close()
        self.connect.close()
        return results

    def change_data_sql(self, sql):
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        self.cursor = self.connect.cursor()
        num = self.cursor.execute(sql)
        self.connect.commit()
        self.cursor.close()
        self.connect.close()
        return num

    def change_data_sqllist(self, sqllist):
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        self.cursor = self.connect.cursor()
        num = 0
        for sql in sqllist:
            num = self.cursor.execute(sql) + num
        self.connect.commit()
        self.connect.close()
        self.connect.close()
        return num

    def get_data_sql(self, sql):

        """
        查询数据
        """
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        df_tel = pd.read_sql(sql, con=self.connect)
        self.connect.close()
        return df_tel

    def get_executemany_sql(self, sql, data_info):
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        self.cursor = self.connect.cursor()
        self.cursor.executemany(sql, data_info)
        self.connect.commit()
        self.cursor.close()
        self.connect.close()

    def get_update_batch_sql(self, sql_list):
        self.connect = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                       db=self.db, charset=self.charset)
        self.cursor = self.connect.cursor()
        for sql in sql_list:
            self.cursor.execute(sql)
        self.connect.commit()
        self.cursor.close()
        self.connect.close()
        print('批量更新完成')
