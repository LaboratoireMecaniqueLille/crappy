#coding: utf-8


from .path import Path

class Constant(Path):
  """
  Simplest condition: will send value unil condition is reached and end
  """
  def __init__(self,time,cmd,condition,value=None):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.value = cmd if value is None else value

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    return self.value
