# coding: utf-8

"""
This example demonstrates the use of the IOBlock Block for reading an InOut in
streamer mode. It does not require any hardware to run, but necessitates the
Python module psutil to be installed.

The IOBlock can interact with hardware connected to the computer. It can read
acquired values, and/or set commands on the device. It interfaces with the
InOut objects of Crappy.

Here, the IOBlock acquires data from a FakeInOut InOut in streamer mode and
sends it to a Grapher Block for display. It is only possible to set the
streamer mode to True because this specific InOut supports it. Stream data is
mostly meant to be saved in Crappy, but that is already demonstrated in the
hdf5_recorder example. Instead, this example demonstrates how to make stream
data readable by any Block using the Demux Modifier.

After starting the script, just watch the memory consumption being displayed
in the Grapher. You can open and close heavy applications (like videos in a web
browser) and watch how the memory usage evolves accordingly. This script must
be stopped by pressing CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This IOBlock drives the FakeInOut InOut, that can read and set the memory
  # usage of the system. Here, it is used in streamer mode and thus returns
  # numpy arrays instead of single data points
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'stream'),  # The names of the labels to output
      streamer=True,  # Reading the InOut in streamer mode
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher displays an overview of the data carried by the stream
  # It can only do so because a Demux Modifier is used on the Link
  graph = crappy.blocks.Grapher(
      # The names of the labels to display
      ('t(s)', 'memory'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  # A Demux Modifier is added to "convert" the stream into data usable by other
  # Blocks, but most of the data is lost in the process
  crappy.link(io, graph,
              modifier=crappy.modifier.Demux(
                  labels='memory',  # The label to associate to the demux data
                  stream_label='stream',  # The label carrying the stream data
                  mean=True,  # Keep only the average of each received chunk

                  # Sticking to default for the other arguments
              ))

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
