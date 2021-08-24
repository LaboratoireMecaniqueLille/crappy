# coding: utf-8

"""
Introduction to the generator.

Shows how to use the Generator block and different paths.

No required hardware.
"""

import crappy

if __name__ == "__main__":
  # First part : a constant value (2) for 5 seconds
  path1 = {'type': 'constant', 'value': 2, 'condition': 'delay=5'}
  # Second: a sine wave of amplitude 1, freq 1Hz for 5 seconds
  path2 = {'type': 'sine', 'amplitude': 1, 'freq': 1, 'condition': 'delay=5'}
  # Third: A ramp rising a 1unit/s until the command reaches 10
  path3 = {'type': 'ramp', 'speed': 1, 'condition': 'cmd>10'}
  # Fourth : cycles of ramps: go down at 1u/s until cmd is <9
  # then go up at 2u/s for 1s. Repeat 5 times
  path4 = {'type': 'cyclic_ramp', 'speed1': -1, 'condition1': 'cmd<9',
           'speed2': 2, 'condition2': 'delay=1', 'cycles': 5}

  # The generator: takes the list of all the paths to be generated
  # cmd_label specifies the name to give the signal
  # freq : the target of points/s
  # spam : Send the value even if nothing changed
  #   (so the graph updates continuously)
  # verbose : add some information in the terminal
  gen = crappy.blocks.Generator([path1, path2, path3, path4],
                                cmd_label='cmd', freq=50, spam=True,
                                verbose=True)

  # The graph : we will plot cmd over time
  graph = crappy.blocks.Grapher(('t(s)', 'cmd'))

  # Do not forget to link them or the graph will have nothing to plot !
  crappy.link(gen, graph)

  # Let's start the program
  crappy.start()
