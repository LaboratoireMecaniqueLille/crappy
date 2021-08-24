# coding: utf-8

"""
Just like ``read.py``, but with both output and input.

This program will send a sine wave on the first output and read the first
input. The type of board can be chosen when starting the file.

Required hardware:
  - Any DAQ board
"""

import crappy

if __name__ == "__main__":
  for i, c in enumerate(crappy.inout.inout_dict):
    print(i, c)
  name = list(crappy.inout.inout_dict.keys())[int(input(
      "What board do you want to use ?> "))]

  sg = crappy.blocks.Generator([{'type': 'sine', 'freq': .5, 'amplitude': 1,
                                 'offset': .5, 'condition': 'delay=1000'}],
                               cmd_label='cmd')

  io = crappy.blocks.IOBlock(name, labels=['t(s)', 'chan0'],
                             cmd_labels=['cmd'], out_channels=0, verbose=True)
  crappy.link(sg, io)

  g = crappy.blocks.Grapher(('t(s)', 'chan0'))

  crappy.link(io, g)

  crappy.start()
