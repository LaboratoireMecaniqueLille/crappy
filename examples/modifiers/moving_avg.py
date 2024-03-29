# coding: utf-8

"""
This example demonstrates the use of the MovingAvg Modifier. It does not
require any specific hardware to run, but necessitates the matplotlib Python
module to be installed.

The MovingAvg Modifier averages all the data it receives from its upstream
Block. Each time a new value is received, it calculates the moving average over
a given number of already received data points, and sends it to the downstream
Block. It therefore preserves the data rate of the signal flowing through the
Link. This Modifier is useful for smoothening a noisy signal, while preserving
its data rate.

Here, a sine wave is generated by a Generator Block and sent to two Grapher
Blocks for display. One Grapher displays it as it is generated, and the other
displays it averaged by a MovingAvg Modifier. Because the frequency of the sine
wave and that of the averaging are the same, the average value as returned by
the MovingAvg Modifier quickly becomes close to 0.

After starting this script, just watch how the raw signal is transformed by the
MovingAvg Modifier and averaged to 0. Also notice how the initial data rate of
the signal is preserved when passing through the MovingAvg Modifier. This demo
ends after 22s. You can also hit CTRL+C to stop it earlier, but it is not a
clean way to stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a sine wave for the Graphers to display. It
  # sends it to one Grapher that displays it as is, and to another Grapher that
  # receives it averaged by the MovingAvg Modifier
  gen = crappy.blocks.Generator(
      # Generating a sine wave of amplitude 2 and frequency 1
      ({'type': 'Sine', 'condition': 'delay=20', 'amplitude': 2, 'freq': 1},),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='sine',  # The label carrying the generated signal

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the raw sine wave it receives from the
  # Generator. As the Generator runs at 30Hz, 30 data points are received each
  # second for display
  graph = crappy.blocks.Grapher(
      ('t(s)', 'sine'),  # The names of the labels to plot on the graph
      interp=False,  # Not linking the displayed spots, to better see the
      # frequency of the input
      length=150,  # Only displaying the data for the last 150 points (~5s)

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the averaged sine wave it receives from the
  # Generator. Because the frequency of the signal is the same as that of the
  # averaging, only values close to 0 are received. Data is received at the
  # same frequency as generated by the Generator, because the MovingAvg
  # Modifier sends an averaged value each time it receives a data point
  graph_avg = crappy.blocks.Grapher(
      ('t(s)', 'sine'),  # The names of the labels to plot on the graph
      interp=False,  # Not linking the displayed spots, to better see the
      # frequency of the input
      length=150,  # Only displaying the data for the last 150 points (~5s)

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)
  crappy.link(gen, graph_avg,
              # Adding a MovingAvg Modifier for calculating the average of the
              # sine wave before sending to the Grapher. For each received data
              # point, an averaged value is sent for display
              modifier=crappy.modifier.MovingAvg(30))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
