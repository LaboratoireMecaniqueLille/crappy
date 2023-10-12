# coding: utf-8

import crappy
from time import time
import numpy as np


class CustomStreamerInOut(crappy.inout.InOut):

  def __init__(self, init_value=0):

    super().__init__()
    self.value1 = init_value
    self.value2 = init_value

  def get_data(self):

    return time(), self.value1, self.value2

  def set_cmd(self, v1, v2):

    self.value1 = v1
    self.value2 = v2

  def get_stream(self):

    t = np.empty((10,))
    val = np.empty((10, 2))

    for i in range(10):
      t[i] = time()
      val[i, 0] = self.value1
      val[i, 1] = self.value2

    return t, val


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

  io = crappy.blocks.IOBlock('CustomStreamerInOut',
                             cmd_labels=('command', 'commandx2'),
                             labels=('t(s)', 'stream'),
                             streamer=True,
                             freq=30)

  graph = crappy.blocks.Grapher(('t(s)', 'val1'), ('t(s)', 'val2'))

  crappy.link(gen, io, modifier=double)
  crappy.link(io, graph,
              modifier=crappy.modifier.Demux(labels=('val1', 'val2')))

  crappy.start()
