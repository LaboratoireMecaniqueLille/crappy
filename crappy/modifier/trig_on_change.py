#coding: utf-8

from .modifier import Modifier


class Trig_on_change(Modifier):
  """
  Can be used to trig an evant when the value of a given label changes.

  Args:
    - name: The name of the label to monitor.

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
