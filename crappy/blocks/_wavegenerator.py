# coding: utf-8

from __future__ import division, print_function

from _masterblock import MasterBlock
import numpy as np
from time import time, sleep


class WaveGenerator(MasterBlock):
  def __init__(self, *args, **kwargs):
    MasterBlock.__init__(self)
    self.waveform = kwargs.get('waveform', 'sin')
    self.nb_points = kwargs.get('nb_points', 50)
    self.wave_frequency = kwargs.get('wave_frequency', 1)
    print('wave_freq:', self.wave_frequency)
    self.labels = ['signal']
    self.gain = kwargs.get('gain', 1)
    self.offset = kwargs.get('offset', 0)
    self.duty_cycle = kwargs.get('duty_cycle', None)

  def sin(self):
    x = np.linspace(0, 2 * np.pi, num=self.nb_points, endpoint=False)
    sin = np.sin(x)
    while True:
      for i in sin:
        self.temporization(self.wave_frequency * self.nb_points)
        yield i * self.gain + self.offset

  def triangle(self):
    x = np.linspace(0, 1, num=self.nb_points, endpoint=True)
    while True:
      for i in x:
        self.temporization(self.wave_frequency * self.nb_points * 2)
        yield i * self.gain + self.offset
      for i in x[::-1]:
        self.temporization(self.wave_frequency * self.nb_points * 2)
        yield i * self.gain + self.offset

  def square(self):
    while True:
      for i in 0, 1:
        self.temporization(self.wave_frequency * 2)
        yield i * self.gain + self.offset

  def pwm(self):
    assert self.duty_cycle, 'Please define a duty cycle.'
    x1 = np.zeros(int((1 - self.duty_cycle) * self.nb_points))
    x2 = np.ones(int(self.duty_cycle * self.nb_points))
    x = np.append(x2, x1)
    while True:
      for i in x:
        self.temporization(self.wave_frequency * self.nb_points)
        yield i * self.gain + self.offset

  def saw_tooth(self):
    x = np.linspace(0, 1, num=self.nb_points, endpoint=True)
    while True:
      for i in x:
        yield i * self.gain + self.offset

  def temporization(self, timeout):
    t_a = time()
    while True:
      sleep(1 / (100 * timeout))
      t_b = time()
      if t_b - t_a >= 1 / timeout:
        break

  def main(self):
    function = getattr(self, self.waveform)()
    while True:
      current = function.next()
      self.send({'signal': current})
