# coding: utf-8

"""
This example demonstrates the use of the Diff Modifier. It does not require any
specific hardware to run, but necessitates the matplotlib Python module to be
installed.

The Diff Modifier calculates the time derivative of a given label, and adds it
to the message to send. It is useful for driving Blocks that need to take
decisions based on the derivative of a signal.

Here, a sine wave is generated by a Generator Block and sent to a Grapher Block
for display. In the Link between these two Blocks, a Diff Modifier calculates
the derivative of the sine wave and adds it to the message. The Grapher
displays both the sine wave and its derivative.

After starting this script, just observe the sine wave and its derivative be
displayed on the graph. You can recognize a nice cosine wave for the derivative
of the sine. This demo never ends, and must be stopped by hitting CTRL+C.
"""

import crappy
from math import pi

if __name__ == '__main__':

  # This Generator generates a sine wave that is sent to the Grapher Block for
  # display. Before that, it is processed by the Diff Modifier, that adds the
  # derivative of the signal to the message. The derivative of the sine wave 
  # is, as everyone knows, a cosine wave.
  gen = crappy.blocks.Generator(
      # Generating a sine wave of amplitude 2 and frequency 1 / (2 * pi)
      # This frequency ensures that its derivative will also have an amplitude
      # of 2
      ({'type': 'Sine', 'amplitude': 2, 'freq': 1 / (2 * pi),
        'condition': None},),
      cmd_label='sine',  # The label carrying the generated signal
      freq=100,  # Using a quite high frequency to get a smooth display

      # Sticking to default for the other arguments
  )

  # This Grapher displays the signal generated by the Generator, as well as its
  # derivative as calculated by the Diff Modifier placed on the Link. Note that 
  # the d_sine label is not generated by the Generator but is added by the Diff 
  # Modifier instead
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'sine'), ('t(s)', 'd_sine'),
      length=500,  # Limiting the extent of the graph to better see the signals

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph,
              # Adding a Diff Modifier for calculating the derivative of the
              # sine wave before sending the data to the Grapher. The 
              # derivative is sent under the d_sine label
              modifier=crappy.modifier.Diff(label='sine', 
                                            out_label='d_sine'))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
