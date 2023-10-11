# coding: utf-8

import crappy
from time import time


def func_modifier(data):

  data['cmdx2'] = data['cmd'] * 2
  return data


class ClassModifier(crappy.modifier.Modifier):

  def __init__(self, label):

    super().__init__()
    self.label = label
    self.sum = 0
    self.last_t = time()

  def __call__(self, data):

    t = time()
    self.sum += data[self.label] * (t - self.last_t)
    self.last_t = t

    data['cumsum'] = self.sum
    return data


if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Sine', 'freq': 0.5,
                                  'amplitude': 2, 'condition': 'delay=15'},),
                                cmd_label='cmd',
                                freq=30)

  graph = crappy.blocks.Grapher(
      ('t(s)', 'cmd'),
      ('t(s)', 'cmdx2'),
      ('t(s)', 'cumsum'),
      )

  crappy.link(gen, graph, modifier=func_modifier)
  crappy.link(gen, graph, modifier=ClassModifier('cmd'))

  crappy.start()
