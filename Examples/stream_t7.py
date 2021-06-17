# coding: utf-8

"""
Demonstrates how to use the stream mode on Labjack T7 boards.

Hardware required:
  - Labjack T7
"""

import crappy
import tables
import numpy as np


def my_mean(data):
  """Average the blocks of data to lower the freq and allow a real-time
  plot."""

  for k, val in data.items():
    data[k] = np.mean(val)
  return data


if __name__ == "__main__":
  s = crappy.blocks.IOBlock("T7_streamer",
      channels=[{'name': 'AIN0', 'gain': 2, 'offset': -13},
        {'name': 'AIN1', 'gain': 2, "make_zero": True}],
      streamer=True)

  g = crappy.blocks.Grapher(('t', 'AIN0'), ('t', 'AIN1'))
  crappy.link(s, g, modifier=my_mean)

  save = crappy.blocks.Hdf_saver("out.h5", atom=tables.Float64Atom())
  crappy.link(s, save)
  crappy.start()
