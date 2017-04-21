#coding: utf-8
from __future__ import print_function

from time import time

from .path import Path

class Cyclic_ramp(Path):
  """
  A "boosted" cyclic path: will take TWO speeds and condtions.
  It will start the first ramp, switch to the second when the first condtion
  is reached and restart the first when the second condtion is reached.
  This will be done "cycles" times (supporting half cycles for ending after
  the first condition), always preserving continuity
  [{'type':'cyclic_ramp','speed1':5,'condition1':'AIN0>2',
  'speed2':-2,'condition2':'AIN1<1','cycles':5}]
  is equivalent to
  [{'type':'ramp','speed':5,'condition':'AIN0>2'},
  {'type':'ramp','value':-2,'condition':'AIN1<1'}]*5
  Note that unlike paths.Constant, paths.Cyclic will ignore previous value
  of cmd and set value1 before any condition is reached
  """
  def __init__(self,time,cmd,condition1,condition2,speed1,speed2,cycles=1,
      verbose=False):
    Path.__init__(self,time,cmd)
    self.speed = (speed1,speed2)
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
      t = time()
      self.cmd += self.speed[self.cycle%2]*(t-self.t0)
      self.t0 = t
      self.cycle +=1
      if self.verbose:
        print("cyclic ramp {}/{}".format(self.cycle,self.cycles))
    return self.speed[self.cycle%2]*(time()-self.t0)+self.cmd
