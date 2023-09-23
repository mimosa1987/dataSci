# -*- coding:utf-8 -*-
import configparser
from datasci.workflow.config.global_config import global_config

# 全局参数变化获取
def get_global_config(section,options):
    conf = configparser.ConfigParser()
    file_dir = global_config.get('db_config')  # 路径自己指定，我这里是以settings.py为参考，abspath是取它的上级目录，也可以直接指定绝对路径来读取
    conf.read(file_dir)  # 读config.ini文件
    value = conf.get(section, options)  # 获取[Mysql-Dse]中host对应的值
    return value


def get_config(filepath=None):
    """
    获得配置文件
    :param filepath: 配置文件路径
    :return:
    """
    if filepath:
        config_path = filepath
    else:
        # 配置文件的绝对路径
        config_path = global_config.get('db_config')
    conf = configparser.ConfigParser()
    conf.read(config_path)
    # 读取配置文件
    return conf
