# coding: utf-8

"""
This example demonstrates the use of a Generator Block to generate an extremely
complex path. It does not require any specific hardware to run, but
necessitates the matplotlib Python module to be installed.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the Generator is used to output a very complex path, creating using list
comprehension and f-strings. The overall shape of this path could have been
generated more smoothly by other means, but the goal here is to show the
potential complexity of the Generator paths.

After starting this script, you can visualize the shape of the generated signal
in the Grapher window. Take a moment to contemplate how twisted the path
instantiation is. This demo never ends, and must be stopped by hitting CTRL+C.
"""

import crappy
from math import acos

if __name__ == '__main__':

  # This horribly complex path is a very badly sampled cosine wave. It could
  # have been generated much more smoothly by creating a custom Generator Path,
  # but it's used here for the sake of the demo
  # The idea is to show how complex and customized the Generator paths can get
  # in Crappy
  path = sum(([{'type': 'Constant',
                'condition': f"""delay={2 * (acos(1 - 2 * (i + 1) / 20) - 
                                             acos(1 - 2 * i / 20))}""",
                'value': 1 - 2 * i / 20} for i in range(20)],
             [{'type': 'Constant',
               'condition': f"""delay={2 * (acos(-1 + 2 * i / 20) - 
                                            acos(-1 + 2 * (i + 1) / 20))}""",
               'value': -1 + 2 * i / 20} for i in range(20)]), list())

  # This Generator Block generates a very complex signal, and sends it to the
  # Grapher Block for display
  # The signal is a very badly sampled cosine wave
  gen = crappy.blocks.Generator(
      path,  # The complex path to generate
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the value of the generated
      # signal
      path_index_label='index',  # This label carries the index of the current
      # path
      repeat=True,  # When reaching the end of the path, repeat it forever
      spam=True,  # Send a value at each loop, for a nice display on the
      # Grapher

      # Sticking to default for the other arguments
      )

  # This Grapher displays the signal it receives from the Generator
  graph = crappy.blocks.Grapher(('t(s)', 'signal'))

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
