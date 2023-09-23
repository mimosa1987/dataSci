# coding:utf8

from abc import ABCMeta, abstractmethod


class Dao(metaclass=ABCMeta):
  """
    Data Access Object Interface.
  """
  def __init__(self, conf):
    self._conf = conf

  @abstractmethod
  def connect(self):
    pass

  @abstractmethod
  def disconnect(self):
    pass
