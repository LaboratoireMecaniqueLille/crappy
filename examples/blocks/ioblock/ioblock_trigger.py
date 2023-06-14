# coding: utf-8

"""
This example demonstrates the use of the IOBlock Block in the case when a
trigger signal controls the moment when data is acquired. It does not require
any hardware to run, but necessitates the Python module psutil to be installed.

The IOBlock can interact with hardware connected to the computer. It can read
acquired values, and/or set commands on the device. It interfaces with the
InOut objects of Crappy.

Here, the IOBlock acquires data from a FakeInOut InOut and sends it to a
Grapher Block for display. Because the trigger_label argument is given, data is
only acquired when receiving values over this label. The trigger signal is
generated by a Button Block, and sent to the IOBlock each time the user clicks
on it.

After starting this script, click on the displayed button and notice how a
value is each time acquired and displayed on the Grapher. Normally, nothing
happens when you stop clicking on the button. To stop this script, you must hit
CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # Whenever this Button is pressed, a value is acquired by the IOBlock
  button = crappy.blocks.Button(
      send_0=True,  # The value 0 is sent at the very beginning of the test, so
      # that on the first click there are already 2 data points to draw
      label='trig',  # The label carrying the trigger signal for the IOBlock
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This IOBlock reads the current memory usage of the system, and sends it to
  # downstream Blocks. This is done by controlling the FakeInOut InOut object
  # It only acquires values when receiving data over the trigger label
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'memory'),  # The names of the labels to output
      trigger_label='trig',  # Only acquiring data when receiving a value over
      # this label
      streamer=False,  # Using the IOBlock in regular mode, not streamer mode
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher displays the memory usage acquired by the IOBlock
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'memory'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(button, io)
  crappy.link(io, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
