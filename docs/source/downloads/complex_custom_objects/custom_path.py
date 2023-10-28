# coding: utf-8

import crappy
from time import time
from scipy.signal import square
from math import pi


class CustomPath(crappy.blocks.generator_path.meta_path.Path):

  def __init__(self,
               low_value,
               high_value,
               duty_cycle,
               freq,
               condition):

    super().__init__()

    # Getting the min and max values, and the frequency
    self._low = min(low_value, high_value)
    self._high = max(low_value, high_value)
    self._freq = freq

    # Initializing the duty cycle attributes
    self._dc_label = None
    self._dc = None

    if isinstance(duty_cycle, str):
      self._dc_label = duty_cycle
    else:
      self._dc = duty_cycle

    # Parsing the given condition
    self._condition = self.parse_condition(condition)

  def get_cmd(self, data):

    # Checking if the stop condition is met
    if self._condition(data):
      raise StopIteration

    # Updating the duty cycle with the latest received value
    if self._dc_label is not None and self._dc_label in data:
      self._dc = data[self._dc_label][-1]

    # If no duty cycle was received over the label, return the low value
    if self._dc is None:
      return self._low

    # Return the scaled signal
    base = square((time() - self.t0) * self._freq * 2 * pi, self._dc)
    return base * (self._high - self._low) / 2 + (self._high + self._low) / 2


if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Sine',
                                  'offset': 0.6,
                                  'amplitude': 0.6,
                                  'freq': 0.25,
                                  'condition': None},),
                                cmd_label='duty',
                                freq=20)

  cust_gen = crappy.blocks.Generator(({'type': 'CustomPath',
                                       'low_value': 5,
                                       'high_value': 15,
                                       'duty_cycle': 'duty',
                                       'freq': 2,
                                       'condition': 'delay=20'},),
                                     cmd_label='square',
                                     spam=True,
                                     freq=50)

  graph = crappy.blocks.Grapher(('t(s)', 'square'))

  crappy.link(gen, cust_gen)
  crappy.link(cust_gen, graph)

  crappy.start()
