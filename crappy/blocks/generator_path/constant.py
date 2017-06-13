#coding: utf-8
from __future__ import print_function

from .path import Path

class Constant(Path):
  """
  Simplest condition. It will send value until condition is reached

  Args:
    value: What value must be sent.

    condition: String representing the condition to end this path.
    See Path.parse_condition for more detail.
  """
  def __init__(self,time,cmd,condition,value=None):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.value = cmd if value is None else value

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    return self.value
