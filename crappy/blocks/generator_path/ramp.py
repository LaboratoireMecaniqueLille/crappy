#coding: utf-8


from time import time

from .path import Path

class Ramp(Path):
  """
  Will make a ramp from previous value until condition is reached,
  speed is in unit/s.
  """
  def __init__(self,time,cmd,condition,speed):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.speed = speed

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    return (time() - self.t0)*self.speed+self.cmd
