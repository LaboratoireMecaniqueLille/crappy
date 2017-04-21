# coding: utf-8
import abc


class Condition:
  """
  Metaclass for all Links conditions. Must implement the evaluate method.
  """

  def __init__(self):
    pass

  __masterblockclass__ = abc.ABCMeta

  @abc.abstractmethod
  def evaluate(self):
    """
    This method is called by the Links and must always be implemented.
    """
    pass
