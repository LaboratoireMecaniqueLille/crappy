#coding: utf-8

from .condition import Condition


class Trig_on_value(Condition):
  """
  Can be used to send data (an empty dict) when the input reached a given value
  Args:
    name: The name of the label to monitor
    values: A list containing the possible values to send the signal
  The condition will trig if data[name] is in values.
  """
  def __init__(self,name,values):
    self.name = name
    self.values = values if isinstance(values,list) else [values]

  def evaluate(self,data):
    if data[self.name] in self.values:
      return data
