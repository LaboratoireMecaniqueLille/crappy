# coding: utf-8

"""
This example demonstrates the use of the IOBlock Block in the case when the
make_zero_delay argument in provided. It does not require any hardware to run,
but necessitates the Python modules psutil and matplotlib to be installed.

The IOBlock can interact with hardware connected to the computer. It can read
acquired values, and/or set commands on the device. It interfaces with the
InOut objects of Crappy.

Here, the IOBLock acquires the current memory usage of the system from the
FakeInOut InOut object, and sends it to a Grapher Block for display. Because
the make_zero_delay argument is provided, the IOBlock acquires data for a few
seconds before the test starts, and takes the average of these acquired value
as the 0 for values acquired during the test. This means that the memory values
sent to downstream Blocks will all be relative to the memory value just before
the test starts.

After starting this script, watch the memory usage of the system being plotted
on the Grapher. It should evolve if you open or close heavy applications, like
videos in browser tabs. Unlike in the other examples of the IOBlock, the memory
values start around 0 because they are all offset due to the make_zero_delay
argument. To stop this demo, you must press CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This IOBlock reads the current memory usage of the system, and sends it to
  # downstream Blocks. This is done by controlling the FakeInOut InOut object
  # Because make_zero_delay is set, it acquires values before the test starts
  # to offset the acquired values to 0 once the test has started
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'memory'),  # The names of the labels to output
      make_zero_delay=2,  # This Block will acquire data for 2 seconds before
      # the test starts, and consider the average of it as its 0
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
  crappy.link(io, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
