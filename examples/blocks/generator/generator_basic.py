# coding: utf-8

"""
This example demonstrates the use of a Generator Block in its simplest use
case. It does not require any specific hardware to run, but necessitates the
matplotlib Python module to be installed.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the Generator first outputs a ramp signal for 10 s, followed by a sine
signal for 10 s. A Grapher Block displays the output signal.

After starting this script, you can visualize the shape of the generated signal
in the Grapher window. This script ends after 22 seconds. You can restart it
with different parameters for the Generator paths. Click the stop button to end
the demo early.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a simple signal and sends it to the Grapher
  # for display
  # The signal is a ramp followed by a sine wave, generated for 10 seconds
  # each
  gen = crappy.blocks.Generator(
      # The path to follow is an iterable of dictionaries, each corresponding
      # to one type of path
      # Each type of individual path has its own mandatory and optional
      # arguments
      # Here, switching to the next path after 10 seconds
      ({'type': 'Ramp',
        'speed': 1,
        'condition': 'delay=10',  # Switching to next after a given delay
        'init_value': 0},
       {'type': 'Sine',
        'freq': 0.5,
        'amplitude': 2,
        'condition': 'delay=10',  # Switching to next after a given delay
        'offset': 10}),
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the value of the generated
      # signal
      path_index_label='index',  # This label carries the index of the current
      # path
      repeat=False,  # When reaching the end of the path, stop the test and do
      # not repeat the path forever
      spam=False,  # Only send a value if it's different from the last sent one
      end_delay=2,  # When the path is exhausted, wait for 2 seconds before
      # ending the test

      # Sticking to default for the other arguments
      )

  # This Grapher displays the signal it receives from the Generator
  graph = crappy.blocks.Grapher(('t(s)', 'signal'))

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)

  # This Block provides a clean way to stop the test before it ends
  stop = crappy.blocks.StopButton()

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
