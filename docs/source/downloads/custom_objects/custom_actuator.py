# coding: utf-8

import crappy
from time import time


class CustomActuator(crappy.actuator.Actuator):

  def __init__(self, init_speed=1) -> None:

    super().__init__()

    self._speed = init_speed
    self._pos = 0
    self._last_t = time()

  def open(self):

    self._last_t = time()

  def set_speed(self, speed):

    self._speed = speed

  def get_speed(self):

    return self._speed

  def get_position(self):

    t = time()
    delta = self._speed * (t - self._last_t)
    self._last_t = t

    self._pos += delta
    return self._pos


if __name__ == '__main__':

  gen = crappy.blocks.Generator(
      ({'type': 'Constant', 'value': 10, 'condition': 'delay=5'},
       {'type': 'Ramp', 'speed': -2, 'condition': 'delay=10'},
       {'type': 'Constant', 'value': -5, 'condition': 'delay=10'},
       {'type': 'Sine', 'freq': 0.5, 'amplitude': 8, 'condition': 'delay=10'}),
      freq=30,
      cmd_label='target(mm/s)',
      spam=True)

  mot = crappy.blocks.Machine(({'type': 'CustomActuator',
                                'mode': 'speed',
                                'cmd_label': 'target(mm/s)',
                                'position_label': 'pos(mm)',
                                'init_speed': 0},),
                              freq=30)

  graph = crappy.blocks.Grapher(('t(s)', 'pos(mm)'))

  crappy.link(gen, graph)
  crappy.link(gen, mot)
  crappy.link(mot, graph)

  crappy.start()
