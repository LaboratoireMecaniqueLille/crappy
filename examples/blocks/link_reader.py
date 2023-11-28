# coding: utf-8

"""
This example demonstrates the use of the LinkReader Block. It does not require
any hardware nor specific Python module to run.

The LinkReader Block is a basic display Block printing each message it receives
from upstream Links in the console. The messages are all printed separately,
they are not grouped together.

Here, a LinkReader Block displays the messages it receives from a Generator
Block generating a cyclic ramp signal. It is mostly useful for debugging, or
when a very basic display is acceptable. Note that in addition, A StopButton
Block allows stopping the script properly without using CTRL+C by clicking on a
button.

After starting this script, just watch how the messages passed to the
LinkReader are being printed in the console. To end this demo, click on the
stop button that appears. You can also hit CTRL+C, but it is not a clean way to
stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a cyclic ramp signal, and the messages it
  # sends via its Link are displayed by the LinkReader Block.
  gen = crappy.blocks.Generator(
      # Generating a cyclic ramp signal, oscillating in a linear way between 0
      # and 1 with a period of 6s
      ({'type': 'CyclicRamp', 'init_value': 0, 'cycles': 0,
        'condition1': 'delay=3', 'condition2': 'delay=3',
        'speed1': 1, 'speed2': -1},),
      cmd_label='value',  # The labels carrying the generated value
      path_index_label='index',  # The label carrying the current Path index
      freq=5,  # Setting a low frequency to avoid spamming the console

      # Sticking to default for the other arguments
  )

  # This LinkReader Block displays in the console the data flowing through the
  # Link pointing to it. Even if several values are received simultaneously,
  # they are still displayed as individual messages in the console
  reader = crappy.blocks.LinkReader(
      name='Reader',  # A name for identifying the Block in the console
      freq=5,  # Useless to set a higher frequency than the labels to display

      # Sticking to default for the other arguments
  )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, reader)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
