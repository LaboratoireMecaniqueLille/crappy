#coding: utf-8

from .condition import Condition

class Derive(Condition):
  """
  Derivation filter

  This will derive the value at label over time. The time label must
  be specified with time='...'
  """
  def __init__(self,label,time='t(s)'):
    Condition.__init__(self)
    self.label = label
    self.t = time
    self.last_t = 0
    self.last_val = 0

  def evaluate(self,data):
    t = data[self.t]
    val = data[self.label]
    data[self.label] = (data[self.label]-self.last_val)/(t-self.last_t)
    self.last_t = t
    self.last_val = val
    return data
