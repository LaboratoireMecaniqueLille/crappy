# coding: utf-8

"""
This example demonstrates the use of a Generator Block whose stop conditions
depend on the value of a label. It does not require any hardware to run.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the Generator is used to output a cyclic ramp signal. The output signal
is displayed by a Grapher Block. The difference with the basic script is that
here the stop condition of the Generator path depends on the value of a label,
whereas in the basic example a delay is given instead.

After starting this script, you can visualize the shape of the generated signal
in the Grapher window. This script ends after 32s, or it can be stopped before
by hitting CTRL+C. You can also restart it with different parameters for the
Generator path.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block outputs a cyclic ramp to the Grapher for display
  # The difference with the basic example is that here the stop condition of
  # the cyclic ramp depends on the value of a label
  # It depends here on the value of the output of the Generator, but it could
  # have been any other label
  # It is an alternative to the stop condition based on a 'delay'
  gen = crappy.blocks.Generator(
      # In this CyclicRamp, the condition for switching to the next slope of
      # the ramp depends on the current output value
      ({'type': 'CyclicRamp',
        'condition1': 'signal>10',  # Depends on the value of 'signal'
        'condition2': 'signal<0',  # Depends on the value of 'signal'
        'speed1': 1,
        'speed2': -1,
        'init_value': 0,
        'cycles': 3},),
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
  # The Generator is linked to itself because it takes decision based on its
  # own output
  crappy.link(gen, gen)
  crappy.link(gen, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
