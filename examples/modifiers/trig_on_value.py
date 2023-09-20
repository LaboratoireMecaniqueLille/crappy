# coding:utf-8

"""
This example demonstrates the use of the TrigOnValue Modifier. It does not
require any specific hardware to run, but necessitates the matplotlib Python
module to be installed.

The TrigOnValue Modifier only lets data pass if the value of a given label is
contained in a given set of allowed values. This Modifier is useful for letting
data pass only in specific situations, or to ensure a specific condition is met
in the data.

Here, a cyclic signal is generated by a Generator Block and sent to two Grapher
Blocks for display. One Grapher displays it as it is generated, and the other
displays it filtered by a TrigOnValue Modifier. Because the TrigOnValue only
lets the value 1 pass, it totally ignores the value -1 that is sent half of the
time.

After starting this script, watch how the TrigOnValue Modifier filters the
data so that values are only sent when the signal is equal to 1. This demo ends
after 32s, but it can be stopped earlier by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a cyclic signal and sends it to the two
  # Graphers for display. To the first Grapher it sends the raw signal, while
  # on the Link to the other a TrigOnValue Modifier filters the data
  gen = crappy.blocks.Generator(
      # Generating a cyclic signal oscillating between 1 and -1 with a period
      # of 6s and stopping after 5 cycles
      ({'type': 'Cyclic',
        'value1': 1,
        'value2': -1,
        'condition1': 'delay=3',
        'condition2': 'delay=3',
        'cycles': 5},),
      cmd_label='cmd',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's just a demo
      spam=True,  # Sending a value at each loop even if it's identical to the
      # previous, otherwise we cannot see well the effect of the TrigOnValue
      # Modifier

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the raw data it receives from the Generator
  # Block. The data rate is that of the Generator, i.e. 30 points per second
  graph = crappy.blocks.Grapher(
      ('t(s)', 'cmd'),  # The names of the labels to plot on the graph
      interp=False,  # Not linking the displayed spots, to better see the
      # frequency of the input

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the data it receives from the Generator Block,
  # filtered by the TrigOnValue Modifier. When the signal switches to -1, the
  # Grapher stops displaying the received values until the signal switches back
  # to 1
  graph_trig = crappy.blocks.Grapher(
      ('t(s)', 'cmd'),  # The names of the labels to plot on the graph
      interp=False,  # Not linking the displayed spots, to better see the
      # frequency of the input

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)
  crappy.link(gen, graph_trig,
              # Adding a TrigOnValue Modifier for filtering the data before it
              # gets sent to the Grapher. A value is passed to the Grapher only
              # if it is contained in the given set of allowed values
              modifier=crappy.modifier.TrigOnValue('cmd', (1,)))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()