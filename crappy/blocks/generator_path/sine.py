#coding: utf-8


from time import time
import numpy as np

from .path import Path


class Sine(Path):
  """
  To generate a sine wave.

  Args:
    amplitude: Amplitude of the sine wave.

    freq: Frequency (Hz) of the sine.

    condition: String representing the condition to end this path.
    See Path.parse_condition for more detail.

    offset: (default=0) offset of the sine.

    phase: (default=0) phase of the sine.
  """
  def __init__(self,time,cmd,condition,freq,amplitude,offset=0,phase=0):
    Path.__init__(self,time,cmd)
    self.condition = self.parse_condition(condition)
    self.amplitude = amplitude/2
    self.offset = offset
    self.phase = phase
    self.k = 2*np.pi*freq

  def get_cmd(self,data):
    if self.condition(data):
      raise StopIteration
    return np.sin((time() - self.t0)*self.k-self.phase)\
        *self.amplitude+self.offset
