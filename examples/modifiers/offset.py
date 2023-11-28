# coding: utf-8

"""
This example demonstrates the use of the Offset Modifier. It does not require
any specific hardware to run, but necessitates the matplotlib Python module to
be installed.

The Offset Modifier simply shifts the data of given labels, by adding a
constant compensation value to them. For each label, the compensation value is
calculated so that the first output point is exactly shifted to a given value.
This Modifier is useful for plotting nicer graphs, or correcting a sensor
offset that might vary.

Here, a Generator Block sends a ramp signal for display to a Grapher Block with
an offset of 10. On the Link, an Offset Modifier is placed that forces the
initial value of the ramp to 0.

After starting this script, just notice how the ramp with an initial offset of
10 is forcibly shifted to 0 offset by the Offset Modifier. This demo ends after
12s. You can also hit CTRL+C to stop it earlier, but it is not a clean way to
stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # This Generator generates a ramp signal sent to the Grapher for display. On
  # the Link to the Grapher, an Offset Modifier shifts the entire signal to a
  # given value, therefore overwriting the offset given here
  gen = crappy.blocks.Generator(
      # Generating a ramp starting from 0 and increasing at a rate of 0.5/s
      # during 10s
      ({'type': 'Ramp',
        'condition': 'delay=10',
        'init_value': 10,  # The offset of the signal is clearly set to 10
        'speed': 0.5},),
      cmd_label='cmd',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's jut a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the data it receives from the Generator,
  # corrected by the Offset Modifier. The original offset of the signal (10) is
  # replaced by the one given in the Offset Modifier (0)
  graph = crappy.blocks.Grapher(
      ('t(s)', 'cmd'),  # The names of the labels to plot on the graph

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph,
              # Adding an Offset Modifier for offsetting the signal before
              # sending it to the Grapher. The initial offset given when
              # defining the Generator Path is overwritten by the new value
              modifier=crappy.modifier.Offset(labels=('cmd',), offsets=(0, )))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
