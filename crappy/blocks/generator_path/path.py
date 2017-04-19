#coding: utf-8

from time import time

class Path(object):
  """
  Parent class for all paths
  """
  def __init__(self,time,cmd):
    self.t0 = time
    self.cmd = cmd

  def get_cmd(self,data):
    return self.cmd

  def parse_condition(self,condition):
    if not isinstance(condition,str):
      return condition
    if '<' in condition:
      var,val = condition.split('<')
      return lambda data:all([i<float(val) for i in data[var]])
    elif '>' in condition:
      var,val = condition.split('>')
      return lambda data:all([i>float(val) for i in data[var]])
    elif condition.startswith('delay'):
      val = float(condition.split('=')[1])
      return lambda data: time()-self.t0>val
    else:
      return lambda data:True
