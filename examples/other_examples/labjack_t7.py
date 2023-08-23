# coding: utf-8

"""
This example demonstrates the use of a Labjack in Crappy for acquiring data
from digital inputs and driving the digital outputs. It is presented here
because we want to promote the use of Labjack equipment, that we use and
appreciate in our laboratory. It is recommended to first read and use the
blocks/ioblock/*.py examples before starting this one. This example requires
the Python module labjack and matplotlib to be installed. It also necessitates
a working Labjack T7.
"""

import crappy

if __name__ == "__main__":

  # This Generator Block generates a ramp signal that drives the digital output
  # of the Labjack
  gen = crappy.blocks.Generator(
      # Generating a linearly increasing signal
      ({"type": "Ramp", "speed": 5 / 60, "condition": None, 'init_value': 0},),
      cmd_label='cmd',  # The label carrying the generated signal
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This IOBlock drives a Labjack acquisition board. It acquires data from the
  # AIN0 and AIN1 ports, and sends it to the Grapher Block for display. It also
  # drives the DAC0 output based on the command received from the Generator
  # Block
  daq = crappy.blocks.IOBlock(
      'LabjackT7',  # The name of the InOut to drive
      # This dictionary contains the information needed to set up the channels
      # to use on the Labjack
      channels=[{'name': 'AIN0', 'gain': 1},
                {'name': 'AIN1', 'gain': 1, 'make_zero': True},
                {'name': 'DAC0', 'gain': 1}],
      labels=['t(s)', 'AIN0', 'AIN1'],  # The labels to send to downstream
      # Blocks carrying the acquired signals
      cmd_labels=['cmd'],  # The label carrying the command to set on DAC0
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays the signals acquired by the Labjack on the AIN0
  # and AIN1 channels
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'AIN0'), ('t(s)', 'AIN1'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, daq)
  crappy.link(daq, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
