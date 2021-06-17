# coding: utf-8

"""
Example demonstrating the use of the generator with steps.

In this example, we would like to reach different levels of strain and relax
the sample (return to `F=0`) between each strain level.

No required hardware.
"""

import crappy

if __name__ == "__main__":
  speed = 5 / 60  # mm/sec

  path = []  # We will put in this list all the paths to be followed

  # We will loop over the values we would like to reach
  # And add two paths for each loop: one for loading and one for unloading
  for exx in [.25, .5, .75, 1., 1.5, 2]:
    path.append({'type': 'constant',
      'value': speed,
      'condition': 'Exx(%)>{}'.format(exx)})  # Go up to this level
    path.append({'type': 'constant',
      'value': -speed,
      'condition': 'F(N)<0'})  # Go down to F=0N

  # Now we can simply give our list of paths to the generator
  generator = crappy.blocks.Generator(path=path)

  # This block will simulate a tensile testing machine
  machine = crappy.blocks.Fake_machine()
  # We must link the generator to the machine to give the command to
  # the machine
  crappy.link(generator, machine)
  # But also the machine to the generator because we added conditions on force
  # and strain, so the generator needs these values coming out of the machine
  # Remember : links are one way only !
  crappy.link(machine, generator)

  # Let's add two graphs to visualise in real time
  graph_def = crappy.blocks.Grapher(('t(s)', 'Exx(%)'))
  crappy.link(machine, graph_def)

  graph_f = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  crappy.link(machine, graph_f)

  # And start the experiment
  crappy.start()
