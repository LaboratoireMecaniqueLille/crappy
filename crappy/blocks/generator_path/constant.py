#coding: utf-8
from __future__ import print_function

from .path import Path

class Constant(Path):
  """
  Simplest condition: will send value unil condition is reached and end
  """
  def __init__(self,time,cmd,condition,value):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.value = value

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    return self.value
