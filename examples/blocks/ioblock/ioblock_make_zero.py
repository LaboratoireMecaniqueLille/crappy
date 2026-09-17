# coding: utf-8

"""
This example demonstrates the use of the IOBlock Block when the
make_zero_delay argument is provided. It does not require any hardware to run,
but necessitates the Python modules psutil and matplotlib to be installed.

The IOBlock can interact with hardware connected to the computer. It can read
acquired values and/or set commands on the device. It interfaces with the
InOut objects of Crappy.

Here, the IOBlock acquires the current memory usage of the system from the
FakeInOut object and sends it to a Grapher Block for display. Because
the make_zero_delay argument is provided, the IOBlock acquires data for a few
seconds before the test starts and uses the average of these values as the zero
reference during the test. The memory values sent to downstream Blocks are
therefore relative to the memory usage measured just before the test starts.

After starting this script, watch the memory usage of the system being plotted
on the Grapher. It should evolve if you open or close heavy applications, like
videos in browser tabs. Unlike in the other IOBlock examples, the memory
values start around zero because they are offset by the make_zero_delay
argument. To end this demo, click on the stop button that appears.
"""

import crappy

if __name__ == '__main__':

  # This IOBlock reads the current memory usage of the system, and sends it to
  # downstream Blocks. This is done by controlling the FakeInOut InOut object
  # Because make_zero_delay is set, it acquires values before the test starts
  # to offset the acquired values to zero once the test has started
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'memory'),  # The names of the labels to output
      make_zero_delay=2,  # This Block will acquire data for 2 seconds before
      # the test starts and considers their average to be zero
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

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(io, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
