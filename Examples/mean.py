# coding: utf-8

"""
Demonstrates the use of the Mean block.

Not to be mistaken with ``modifier.Mean`` ! The Mean blocks performs the mean
at a given frequency, whereas the modifier will perform the average over a
chosen number of points regardless of the frequency.

No required hardware.
"""

import crappy

if __name__ == "__main__":
  g1 = crappy.blocks.Generator(
      [dict(type='sine', freq=2, amplitude=2, condition=None)],
      freq=200,
      cmd_label='cmd1'
    )

  g2 = crappy.blocks.Generator(
      [dict(type='sine', freq=.2, amplitude=2, condition=None)],
      freq=200,
      cmd_label='cmd2'
    )

  # Return a point every .5s
  # It will be the average received value
  m = crappy.blocks.Mean_block(.5)  # , out_labels=['cmd1', 'cmd2'])

  crappy.link(g1, m)
  crappy.link(g2, m)

  g = crappy.blocks.Grapher(('t(s)', 'cmd1'), ('t(s)', 'cmd2'))
  crappy.link(m, g)
  crappy.start()
