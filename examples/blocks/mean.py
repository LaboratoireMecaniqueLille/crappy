# coding: utf-8

"""
This example demonstrates the use of the Mean Block. It does not require any
specific hardware to run, but necessitates the matplotlib Python module to be
installed.

The Mean Block averages the data it receives over a given period of time, and
sends the averaged data to the downstream Blocks. Its behavior is similar to
that of the Mean and MovingAvg Modifiers, except these average data over a
given number of points.

Here, the Mean Block averages the data it receives from two Generator Blocks,
and sends the averaged data to a Grapher Block for display. Because the
averaging period is equal to the period of the averaged signals, the output
of the Mean Block is just the offset of the signals.

After starting the script, wait a few seconds for the average values to be
calculated. Then, a graph appears on which you should normally only see the
offsets of the signals. This script never ends, and must be stopped by hitting
CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a sine wave and sends it to the Mean Block
  # for averaging
  gen_1 = crappy.blocks.Generator(
      # Generating a sine wave of frequency 2, offset -1, and amplitude 1
      ({'type': 'Sine', 'freq': 2, 'amplitude': 1, 'condition': None,
        'offset': -1},),
      cmd_label='label_1',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Generator Block generates a sine wave and sends it to the Mean Block
  # for averaging
  gen_2 = crappy.blocks.Generator(
      # Generating a sine wave of frequency 2, offset 1, and amplitude 1
      ({'type': 'Sine', 'freq': 2, 'amplitude': 1, 'condition': None,
        'offset': 1},),
      cmd_label='label_2',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Mean Block averages the data it receives from the two Generator Blocks
  # Because the period of these two signals is the same as the averaging
  # period, only the average of the signals is transmitted to the Grapher for
  # display
  mean = crappy.blocks.MeanBlock(
      delay=2,  # The time period over which to perform the averaging
      out_labels=('label_1', 'label_2'),  # The labels to perform the averaging
      # on, in addition to the time label
      freq=20,  # This frequency does not matter as the Block doesn't loop
      # while it's averaging the data

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the averaged data it receives from the Mean
  # Block. Because the averaging period is the same as the period of the sine
  # signals, only their offset is left for display on the graph.
  graph = crappy.blocks.Grapher(
      # The names of the labels to display on the graph
      ('t(s)', 'label_1'), ('t(s)', 'label_2')

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen_1, mean)
  crappy.link(gen_2, mean)
  crappy.link(mean, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
