# coding: utf-8

"""
Example demonstrating the use of thermocouples with the Labjack T7 board.

required hardware:
  - Labjack T7 board
  - Thermocouple(s)
"""

import crappy

chan = [0, 2, 3, 4, 5]

if __name__ == "__main__":
  m = crappy.blocks.IOBlock("Labjack_t7",
                            channels=[dict(name='AIN%d' % i, thermocouple='K')
                                      for i in chan],
                            verbose=True, labels=['t(s)'] + ['T%d' % i
                                                             for i in chan])

  g = crappy.blocks.Grapher(*[('t(s)', 'T%d' % i) for i in chan])

  crappy.link(m, g, modifier=crappy.modifier.Moving_avg(10))

  crappy.start()
