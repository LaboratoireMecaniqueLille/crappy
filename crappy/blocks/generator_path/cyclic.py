#coding: utf-8
from __future__ import print_function,division

from time import time

from .path import Path

class Cyclic(Path):
  """
  A "boosted" constant path: will take TWO values and condtions.
  It will set the first value, switch to the second when the first condtion
  is reached and return to the first when the second condtion is reached.
  This will be done "cycles" times (supporting half cycles for ending after
  the first condition)
  [{'type':'cyclic','value1':1,'condition1':'AIN0>2',
  'value2':0,'condition2':'AIN1<1','cycles':5}]
  is equivalent to
  [{'type':'constant','value':1,'condition':'AIN0>2'},
  {'type':'constant','value':0,'condition':'AIN1<1'}]*5
  cycles = 0 will make it loop forever
  """
  def __init__(self,time,cmd,condition1,condition2,value1,value2,cycles=1,
      verbose=False):
    Path.__init__(self,time,cmd)
    self.value = (value1,value2)
    self.condition1 = self.parse_condition(condition1)
    self.condition2 = self.parse_condition(condition2)
    self.cycles = int(2*cycles) # Logic in this class will be in half-cycle
    self.cycle = 0
    self.verbose = verbose

  def get_cmd(self,data):
    if self.cycles > 0 and self.cycle >= self.cycles:
      raise StopIteration
    if not self.cycle % 2 and self.condition1(data) or\
        self.cycle % 2 and self.condition2(data):
      self.cycle +=1
      if self.verbose:
        print("cyclic path {}/{}".format(self.cycle,self.cycles))
      self.t0 = time()
    return self.value[self.cycle%2]
