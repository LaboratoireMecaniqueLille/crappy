#coding: utf-8


from .path import Path


class Constant(Path):
  """
  Simplest condition. It will send value until condition is reached.

  Args:
    - value: What value must be sent.
    - condition (str): Representing the condition to end this path.
      See Path.parse_condition for more detail.
    - send_one: If True, this condition will send the value at least once
      before checking the condition.

  """
  def __init__(self,time,cmd,condition,send_one=True,value=None):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.value = cmd if value is None else value
    if send_one:
      self.get_cmd = self.get_cmd_first
    else:
      self.get_cmd = self.get_cmd_condition

  def get_cmd_first(self,data):
    self.get_cmd = self.get_cmd_condition
    return self.value

  def get_cmd_condition(self,data):
    if self.condition(data):
      raise StopIteration
    return self.value
