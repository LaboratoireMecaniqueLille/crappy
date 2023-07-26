# coding: utf-8

"""
This example demonstrates the use of the IOBlock Block in a basic use case. It
does not require any hardware to run, but necessitates the Python modules
psutil and matplotlib to be installed.

The IOBlock can interact with hardware connected to the computer. It can read
acquired values, and/or set commands on the device. It interfaces with the
InOut objects of Crappy.

Here, the IOBlock drives the FakeInOut InOut that can read and/or adjust the
memory usage of the system. A Generator generates a sinusoidal memory usage
target, and the IOBlock drives the FakeInOut so that the actual memory usage
stays as close as possible to that target. A Grapher Block displays the target
and actual values.

After starting this script, just watch how the actual memory usage is being
driven to match the target. You can monitor the memory usage of the computer
independently using the Task Manager (Windows) or htop (Linux), it should match
with the one displayed in Crappy. Depending on your memory level at the moment
when you start this script, you might need to adjust the offset of the target
to be able to see the effects of this script. For ending this script, press
CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator generates the target memory value to reach
  # The target is a sine wave of amplitude 20 and frequency 0.02 Hz
  gen = crappy.blocks.Generator(
      # The sine wave to generate
      ({'type': 'Sine',
        'amplitude': 20,
        'offset': 50,
        'freq': 0.02,
        'condition': None},),
      freq=30,  # Lowering the default frequency because it's just a demo
      cmd_label='target',  # The label carrying the target memory value

      # Sticking to default for the other arguments
  )

  # This IOBlock reads the current memory usage of the system, and sends it to
  # downstream Blocks
  # It also tries to adjust the memory usage so that it matches the commands it
  # received
  # This is all done by controlling the FakeInOut InOut object
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'memory'),  # The names of the labels to output
      cmd_labels='target',  # The name of the label carrying the target memory
      # usage value
      streamer=False,  # Using the IOBlock in regular mode, not streamer mode
      freq=30,  # Lowering the default frequency because it's just a demo
      spam=False,  # Only setting a command if it's different from the last
      # received one

      # Sticking to default for the other arguments
  )

  # This Grapher displays the target and the actual memory values on a same
  # graph
  graph = crappy.blocks.Grapher(
      # Providing the labels to display
      ('t(s)', 'target'), ('t(s)', 'memory')

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, io)
  crappy.link(gen, graph)
  crappy.link(io, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
