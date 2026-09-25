# coding: utf-8

"""
This example demonstrates the use of a Generator Block, and in particular how
this Block manages the cycles. It does not require any specific hardware to
run, but necessitates the pyqtgraph and PyQt6 Python modules to be installed.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the Generator is used to output cyclic constant signals. The given path
first oscillates three times between 1 and -1, then three times between 2 and
-2. When the path is exhausted, it repeats and loops forever. This example
demonstrates how the path_index_label tracks the index of the current path,
even when the repeat argument is set to True.

After starting this script, you can visualize the shape of the generated signal
in the Grapher window, and watch how the 'index' label is updated each time the
Generator switches to a new path. To end this demo, click on the stop button
that appears.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block outputs a cyclic signal oscillating first between 1
  # and -1 and then between 2 and -2. The given path ends after 3 cycles of
  # each, but it is then repeated. The path_index_label shows which step of the
  # overall path is currently being output
  gen = crappy.blocks.Generator(
      # The cyclic paths oscillate three times between 1 and -1 and then three
      # times between 2 and -2. The exhausted path repeats because 'repeat' is
      # set to True
      ({'type': 'Cyclic',
        'condition1': 'delay=1',
        'condition2': 'delay=1',
        'value1': 1,
        'value2': -1,
        'cycles': 3},
       {'type': 'Cyclic',
        'condition1': 'delay=1',
        'condition2': 'delay=1',
        'value1': 2,
        'value2': -2,
        'cycles': 3}
       ),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the value of the generated
      # signal
      path_index_label='index',  # This label carries the index of the current
      # path
      repeat=True,  # When reaching the end of the path, loop endlessly
      spam=True,  # Send a value at each loop to plot smooth graphs even when
      # the output value does not change

      # Sticking to default for the other arguments
  )

  # This Grapher displays the signal it receives from the Generator, along with
  # the index of the current path
  graph = crappy.blocks.Grapher(('t(s)', 'signal'), ('t(s)', 'index'))

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  # The Generator is linked to itself because it makes decisions based on its
  # own output
  crappy.link(gen, gen)
  crappy.link(gen, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
