# coding: utf-8

"""
Short code to read a channel on a chosen DAQ board.

Will show the supported boards and ask the user the picked one. Then reads
continuously the first channel.
Note that no additional argument is specified to the DAQ board, so the channel,
the rate and precision are the default values.

Required hardware:
  - Any supported DAQ board
"""

import crappy

if __name__ == "__main__":
  lst = list(crappy.inout.in_dict.keys())
  for i, c in enumerate(lst):
    print(i, c)
  name = lst[int(input("What board do you want to use ?> "))]

  m = crappy.blocks.IOBlock(name, labels=['t(s)', 'chan0'], verbose=True)

  g = crappy.blocks.Grapher(('t(s)', 'chan0'))

  crappy.link(m, g)

  crappy.start()
