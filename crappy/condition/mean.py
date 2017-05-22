#coding: utf-8

import numpy as np

from .condition import Condition

class Mean(Condition):
  """
  Mean filter:
    returns the mean value every npoints point of data
  Arg:
    npoints (int): the number of points it takes to return 1 value
  Will divide the output freq by npoints
  If you need the same freq, see Moving_avg
  """
  def __init__(self,npoints=100):
    Condition.__init__(self)
    self.npoints = npoints

  def evaluate(self,data):
    if not hasattr(self,"last"):
      self.last = data
      for k in data:
        self.last[k] = [self.last[k]]
      return
    r = {}
    for k in data:
      self.last[k].append(data[k])
      if len(self.last[k]) == self.npoints:
        r[k] = np.mean(self.last[k])
      elif len(self.last[k]) > self.npoints:
        self.last[k] = []
    if r:
      return r

