# coding: utf-8

from __future__ import division, print_function

from _masterblock import MasterBlock
import numpy as np
from time import time, sleep


class WaveGenerator(MasterBlock):
  def __init__(self, *args, **kwargs):
    MasterBlock.__init__(self)
    self.waveform = kwargs.get('waveform', 'sin')
    self.nb_points = 100
    self.x = np.linspace(0, 2 * np.pi, num=self.nb_points, endpoint=False)
    self.wave_frequency = kwargs.get('wave_frequency', 10)
    self.labels = ['signal']
    self.repeat = kwargs.get('repeat', True)
    self.gain = kwargs.get('gain', 1)
    self.offset = kwargs.get('offset', 0)

  def sin(self):
    sin = np.sin(self.x)

    while True:
      for i in sin:
        yield i

  def triangle(self):
    pass

  def square(self):
    pass

  def saw_tooth(self):
    pass

  def temporization(self):
    t_a = time()
    while time() - t_a < 1 / (self.nb_points * self.wave_frequency):
      # sleep(1 / (10 * self.nb_points * self.wave_frequency))
      pass

  def main(self):
    function = self.sin()
    while True:
      current = function.next()
      # self.temporization()
      self.send({'signal': current})
