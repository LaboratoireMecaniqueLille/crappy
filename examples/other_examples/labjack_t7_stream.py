# coding: utf-8

"""
Demonstrates how to use the stream mode on Labjack T7 boards.

Hardware required:
  - Labjack T7
"""

import crappy
import tables


if __name__ == "__main__":
  s = crappy.blocks.IOBlock("T7Streamer",
                            channels=[{'name': 'AIN0', 'gain': 2,
                                       'offset': -13},
                                      {'name': 'AIN1', 'gain': 2,
                                       "make_zero": True}],
                            streamer=True)

  g = crappy.blocks.Grapher(('t(s)', 'AIN0'), ('t(s)', 'AIN1'))
  crappy.link(s, g, modifier=crappy.modifier.Demux(labels=('AIN0', 'AIN1')))

  rec = crappy.blocks.HDFRecorder("out.h5", atom=tables.Float64Atom())
  crappy.link(s, rec)
  crappy.start()
