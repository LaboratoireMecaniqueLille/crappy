# coding: utf-8

"""
This example demonstrates the use of the Median Modifier. It does not require
any specific hardware to run, but necessitates the matplotlib Python module to
be installed.

The Median Modifier applies a median filter to all the data it receives from
its upstream Block. Once it has gathered enough data points, it sends the
calculated median values to the downstream Block and enters the next processing
loop. It therefore reduces the data rate on a given Link. This Modifier is
useful for reducing the data rate in a given Link while still preserving as
much information as possible.

Here, a sine wave is generated by a Generator Block and sent to two Grapher
Blocks for display. One Grapher displays it as it is generated, and the other
displays it filtered by a Median Modifier. Because the frequency of the sine
wave and that of the filter are the same, the filtered value as returned by the
Median Modifier is always close to 0.

After starting this script, just watch how the raw signal is transformed by the
Median Modifier and filtered to 0. Also notice how the initial data rate of the
signal is divided when passing through the Median Modifier. This demo ends
after 22s. You can also hit CTRL+C to stop it earlier, but it is not a clean
way to stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a sine wave for the Graphers to display. It
  # sends it to one Grapher that displays it as is, and to another Grapher that
  # receives it filtered by the Median Modifier
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

  # This Grapher Block displays the median sine wave it receives from the
  # Generator. Because the frequency of the signal is the same as that of the
  # median filter, only values close to 0 are received. A point is received
  # once every second, because the Median Modifier only outputs data once every
  # 30 received points
  graph_med = crappy.blocks.Grapher(
      ('t(s)', 'sine'),  # The names of the labels to plot on the graph
      interp=False,  # Not linking the displayed spots, to better see the
      # frequency of the input
      length=5,  # Only displaying the data for the last 5 points (~5s)

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)
  crappy.link(gen, graph_med,
              # Adding a Median Modifier for calculating the median of the sine
              # wave before sending to the Grapher. A data point is sent for
              # display once every 30 received points
              modifier=crappy.modifier.Median(30))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
