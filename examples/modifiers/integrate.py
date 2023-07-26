# coding: utf-8

"""
This example demonstrates the use of the Integrate Modifier. It does not
require any specific hardware to run, but necessitates the matplotlib Python
module to be installed.

The Integrate Modifier calculates the time integral of a given label, and adds
it to the message to send. It is useful for driving Blocks that need to take
decisions based on the integral of a signal.

Here, a sine wave is generated by a Generator Block and sent to a Grapher Block
for display. In the Link between these two Blocks, an Integrate Modifier
calculates the integral of the sine wave and adds it to the message. The
Grapher displays both the sine wave and its integral.

After starting this script, just observe the sine wave and its integral be
displayed on the graph. You can recognize a nice 1 - cos(x) wave for the
integral of the sine. This demo never ends, and must be stopped by hitting
CTRL+C.
"""

import crappy
from math import pi

if __name__ == '__main__':

  # This Generator generates a sine wave that is sent to the Grapher Block for
  # display. Before that, it is processed by the Integrate Modifier, that adds
  # the integral of the signal to the message. The integral of the sine wave
  # is, as everyone knows, a cosine wave.
  gen = crappy.blocks.Generator(
      # Generating a sine wave of amplitude 2 and frequency 1 / (2 * pi)
      # This frequency ensures that its integral will also have an amplitude
      # of 2
      ({'type': 'Sine', 'condition': None, 'freq': 1 / (2 * pi),
        'amplitude': 2},),
      cmd_label='sine',  # The label carrying the generated signal
      freq=100,  # Using a quite high frequency to get a smooth display

      # Sticking to default for the other arguments
  )

  # This Grapher displays the signal generated by the Generator, as well as its
  # integral as calculated by the Integrate Modifier placed on the Link. Note
  # that the i_sine label is not generated by the Generator but is added by the
  # Integrate Modifier instead
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'sine'), ('t(s)', 'i_sine'),
      length=500,  # Limiting the extent of the graph to better see the signals

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph,
              # Adding an Integrate Modifier for calculating the integral of
              # the sine wave before sending the data to the Grapher. The
              # derivative is sent under the i_sine label
              modifier=crappy.modifier.Integrate(label='sine',
                                                 out_label='i_sine'))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
