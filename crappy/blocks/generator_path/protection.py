#coding: utf-8


from .path import Path

class Protection(Path):
  """
  Useful to protect samples from being pulled appart when setting up a test

  by default will send value0.
  While condition1 is met, will return value1.
  While condition2 is met, will return value2.
  If condition1 and condition2 are met simultaneaously, the first one met will
  prevail. If met at the same time, condition1 will prevail.
  Args:
    value0: value to send when no condition is reached.

    condition1: String representing the first condition.
    See Path.parse_condition for more detail.

    value1: value to send when condition1 is met

    condition2: String representing the second condition.

    value2: value to send when condition2 is met
  """
  def __init__(self,time,cmd,condition1,condition2,value1,value2,value0=0,
      verbose=False):
    Path.__init__(self,time,cmd)
    self.value = (value0,value1,value2)
    self.condition1 = self.parse_condition(condition1)
    self.condition2 = self.parse_condition(condition2)
    self.verbose = verbose
    self.status = 0

  def get_cmd(self,data):
    if self.status == 0:
      if self.condition1(data):
        self.status = 1
      elif self.condition2(data):
        self.status = 2
      return self.value[self.status]
    if self.status == 1 and not self.condition1(data):
      self.status = 0
    elif self.status == 2 and not self.condition2(data):
      self.status = 0
    return self.value[self.status]

