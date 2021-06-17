# coding: utf-8

"""
A fully autonomous example of a tensile test using a virtual machine.

This program is meant to replicate a tensile test without the need for an
actual machine.

No hardware required.
"""

import crappy

if __name__ == "__main__":
  speed = 5/60  # mm/sec

  generator = crappy.blocks.Generator(path=sum([[
    {'type': 'constant', 'value': speed,
     'condition': 'Exx(%)>{}'.format(i / 3)},
    {'type': 'constant', 'value': -speed, 'condition': 'F(N)<0'}]
    for i in range(5)], []), spam=False, cmd_label='cmd')

  machine = crappy.blocks.Fake_machine()

  crappy.link(generator, machine)
  crappy.link(machine, generator)

  # graph_def = crappy.blocks.Grapher(('t(s)', 'Exx(%)'))
  # crappy.link(machine, graph_def)

  graph_f = crappy.blocks.Grapher(('Exx(%)', 'F(N)'))
  crappy.link(machine, graph_f)

  # graph_x = crappy.blocks.Grapher(('t(s)', 'x(mm)'))
  # crappy.link(machine, graph_x)

  crappy.start()
