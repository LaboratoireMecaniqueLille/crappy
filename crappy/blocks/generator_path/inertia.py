#coding: utf-8
from __future__ import print_function,division
import numpy as np

from .path import Path


class Inertia(Path):
  """
  Used to lower/higher the output command by integrating an input over time.

  Let f(t) be the input signal, v(t) the value of the output,
  m the inertia and t0 the beginning of this path. K is a choosen constant.

  Note:
    The output value for this path will be:
    v(t) = v(t0) - K*[I(t0->t)f(t)dt]/m

  Args:
    - inertia (float): This is the virtual inertia of the process.

      Note:
        The higher it is, the slower the (in/de)crease will be.

        In the above formula, it is the value of m.

    - flabel (str): The name of the label of the value to integrate.
    - tlabel (str, default: 't(s)'): The name of the label of time for the
      integration.

      Note:
        The data received by flabel and tlabel must correspond.

        In other word, there must be exactly the same number of values
        received by these two labels at any instant
        (ie they must come from the same parent block).

    - const (float, default: 30/pi): The value of K in the formula above.

      Note:
        The default value is meant to send rpm with inertia in kgm² and torque
        in Nm.

        If sending rad/s, use const=1.

    - condition (str): Condition to meet to end this path.
      See :ref:`generator path` for more info.

  """
  def __init__(self,time,cmd,condition,inertia,
      flabel,const=30/np.pi,tlabel='t(s)',value=None):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.inertia = inertia
    self.flabel = flabel
    self.tlabel = tlabel
    self.const = const/self.inertia
    self.value = cmd if value is None else value
    self.last_t = None

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    if data[self.tlabel]:
      if self.last_t is None:
        # If it is the first call, we cannot use the first data point, since we
        # don't have the previous time (I use left rectangle integration)
        t = data[self.tlabel]
        if len(t) == 1:
          # If we have only one point, save it and return,
          # first value will be returned on the next call
          self.last_t = t[0]
          return self.value
        # else: drop the first point and keep going
        f = np.array(data[self.flabel][1:])
      else:
        t = [self.last_t]+data[self.tlabel] # We have a previous point: use it
        f = np.array(data[self.flabel])
      dt = np.array([j-i for i,j in zip(t[:-1],t[1:])])
      self.value -= self.const*sum(dt*f) # The actual integration
      self.last_t = t[-1]
    return self.value
