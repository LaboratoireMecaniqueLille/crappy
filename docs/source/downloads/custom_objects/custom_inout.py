# coding: utf-8

import crappy
from time import time


class CustomInOut(crappy.inout.InOut):

  def __init__(self, init_value=0):

    super().__init__()
    self.value1 = init_value
    self.value2 = init_value

  def get_data(self):

    return time(), self.value1, self.value2

  def set_cmd(self, v1, v2):

    self.value1 = v1
    self.value2 = v2


def double(dic):
  dic['commandx2'] = 2 * dic['command']
  return dic


if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Sine',
                                  'amplitude': 2,
                                  'freq': 0.5,
                                  'condition': 'delay=20'},),
                                cmd_label='command',
                                freq=30)

  io = crappy.blocks.IOBlock('CustomInOut',
                             cmd_labels=('command', 'commandx2'),
                             labels=('t(s)', 'val1', 'val2'),
                             freq=30)

  graph = crappy.blocks.Grapher(('t(s)', 'val1'), ('t(s)', 'val2'))

  crappy.link(gen, io, modifier=double)
  crappy.link(io, graph)

  crappy.start()
