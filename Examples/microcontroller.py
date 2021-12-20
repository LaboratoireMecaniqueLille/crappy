# coding: utf-8

"""Example showing the use of the UController block.

This example is meant to be used in combination with the microcontroller.py
template in the tools of Crappy.

Required hardware:
  - A microcontroller

"""

import crappy

if __name__ == "__main__":

  gen = crappy.blocks.Generator([{'type': 'constant',
                                  'value': i,
                                  'condition': 'delay=10'}
                                 for i in range(1, 6)],
                                cmd=1,
                                cmd_label='freq',
                                freq=50)

  uc = crappy.blocks.UController(cmd_labels=['freq'], labels=['nr'],
                                 init_output={'nr': 0}, freq=50)

  graph = crappy.blocks.Dashboard(labels=['t(s)', 'nr'])

  crappy.link(gen, uc)
  crappy.link(uc, graph)

  crappy.start()
