#coding: utf-8

from .condition import Condition


class Trig_on_change(Condition):
  """
  Can be used to trig an evant when the value of a given label changes
  Args:
    name: the name of the label to monitor
  """
  def __init__(self,name):
    self.name = name

  def evaluate(self,data):
    if not hasattr(self,'last'):
      self.last = data[self.name]
      return data
    if data[self.name] == self.last:
      return None
    self.last = data[self.name]
    return data
