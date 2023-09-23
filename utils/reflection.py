# -*- coding: utf-8 -*-
###
# author: zhaolihan
# date	: 20200930
###
import importlib

class Reflection(object):
    @staticmethod
    def reflect_obj(module_path, class_name, params):
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            obj_cls = getattr(module, class_name)
            if params is None or params == "":
                return obj_cls()
            else:
                return obj_cls(**params)
        else:
            raise AttributeError('%s Not have class %s in it!' % (module_path, class_name))

    @staticmethod
    def reflect_func(class_obj, func_name):
        if hasattr(class_obj, func_name):
            func = getattr(class_obj, func_name)
            return func
        else:
            raise AttributeError('%s Not have class %s in it!' % (class_obj, func_name))

    @staticmethod
    def reflect_obj_func(module_path, func_name):
        module = importlib.import_module(module_path)
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            return func
