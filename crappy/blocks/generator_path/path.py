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
    """
    This method turns a string into a fonction that returns a bool
    It is meant to check if a skip condition is reached
    The following syntaxe is supported:
      myvar>myvalue
      myvar<myvalue
    myvar is the label of an input value
    myvalue is a float
    This will return True when the data under the label myvar will be
    larger/smaller than myvalue
      delay=mydelay
    The condtion will turn True after mydelay seconds
    Any other syntax will return True instantly
    """
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
