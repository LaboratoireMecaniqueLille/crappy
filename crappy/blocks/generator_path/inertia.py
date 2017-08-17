#coding: utf-8
from __future__ import print_function,division
import numpy as np

from .path import Path

class Inertia(Path):
  """
  Used to lower/higher the output command by integrating an input over time

  Let f(t) be the input signal, v(t) the value of the output,
  m the inertia and t0 the beginning of this path.
  The output value for this path will be:
  v(t) = v(t0) - [I(t0->t)f(t)dt]/m

  Args:
  inertia (float):
    This is the virtual inertia of the process.
    The higher it is, the slower the (in/de)crease will be.
    In the above formula, it is the value of m
  flabel (str):
    The name of the label of the value to integrate
  tlabel (str), default is 't(s)':
    The name of the label of time for the integration
    Note: The data received by flabel and tlabel must correspond.
      In other word, there must be exactly the same number of values
      received by these two labels at any instant
      (ie they must come from the same parent block).
  condition (str, see Path for more info):
    Condition to meet to end this path
  """
  def __init__(self,time,cmd,condition,inertia,flabel,tlabel='t(s)',value=None):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.inertia = inertia
    self.flabel = flabel
    self.tlabel = tlabel
    self.value = cmd if value is None else value
    self.last_t = 0

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    if data[self.tlabel]:
      t = [self.last_t]+data[self.tlabel]
      dt = np.array([j-i for i,j in zip(t[:-1],t[1:])])
      f = np.array(data[self.flabel])
      self.value -= sum(dt*f)/self.inertia
      self.last_t = t[-1]
    return self.value
