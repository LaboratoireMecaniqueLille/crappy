# coding: utf-8

"""
This example demonstrates the use of a Generator Block using a user-defined
condition to determine whether to switch to the next Path. It does not require
any specific hardware to run, but necessitates the matplotlib Python module to
be installed.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the Generator outputs a simple constant signal, that switches to a
different value once the end condition is met. However, unlike the other
Generator example, the stop condition is not one of the standard ones defined
in Crappy, but rather an arbitrary callable defined by the user. Here, the
condition check whether a given file exists, but it could really have been any
other kind of condition.

After starting this script, you should create the file 'test.txt' in the same
folder where this script is located. See how the value of the signal changes
once the file is created. Once you delete the newly created file, the test
should then end, due to the second custom condition. You can also end this demo
earlier by clicking on the stop button that appears. You can also hit CTRL+C,
but it is not a clean way to stop Crappy.
"""

import crappy
from pathlib import Path


def file_exists(data):
  """Returns True if the file 'test.txt' exists at the same level as the
  running script, False otherwise.

  This arbitrary function can access the data received by the Generator Block,
  which is exposed in the data argument as a dictionary.

  Args:
    data: The data received by the Generator Block since its last loop. The
      keys are the labels, and the values a list containing all the received
      values for the given label.
  """

  return Path('./test.txt').exists()


def file_does_not_exist(data):
  """Returns False if the file 'test.txt' exists at the same level as the
  running script, True otherwise.

  This arbitrary function can access the data received by the Generator Block,
  which is exposed in the data argument as a dictionary.

  Args:
    data: The data received by the Generator Block since its last loop. The
      keys are the labels, and the values a list containing all the received
      values for the given label.
  """

  return not Path('./test.txt').exists()


if __name__ == '__main__':

  # This Generator Block generates a constant signal, and sends it to the
  # Dashboard Block for display
  # The signal first has a value of 0, then 1.
  gen = crappy.blocks.Generator(
      path=({'type': 'Constant', 'value': 0,
             'condition': file_exists},
            {'type': 'Constant', 'value': 1,
             'condition': file_does_not_exist}),
      # The simple path to generate
      # Notice how the functions defined earlier are included in the path and
      # associated to the 'condition' key
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the value of the generated
      # signal
      path_index_label='index',  # This label carries the index of the current
      # path
      spam=True,  # Send a value at each loop, for a nice display on the
      # Dashboard

      # Sticking to default for the other arguments
      )

  # This Dashboard displays the signal it receives from the Generator
  dash = crappy.blocks.Dashboard(('t(s)', 'signal'))

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, dash)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
