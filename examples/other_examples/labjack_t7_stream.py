# coding: utf-8

"""
This example demonstrates the use of a Labjack in Crappy for acquiring data at
high rates in streamer mode. It is presented here because we want to promote
the use of Labjack equipment, that we use and appreciate in our laboratory. It
is recommended to first read and use the blocks/ioblock/ioblock_streamer.py and
the blocks/hdf5_recorder.py examples before starting this one. This example
requires the Python modules tables, labjack and matplotlib to be installed. It
also necessitates a working Labjack T7.
"""

import crappy
import tables

if __name__ == "__main__":

  # This IOBlock drives a Labjack in streamer mode, and acquires data from the
  # AIN0 and AIN1 ports. It then sends this data to a Grapher Block for display
  # and an HDFRecorder Block for recording
  labjack = crappy.blocks.IOBlock(
      "T7Streamer",  # The name of the InOut to acquire data from
      # This dictionary contains the information needed to set up the channels
      # to use on the Labjack
      channels=[{'name': 'AIN0', 'gain': 2, 'offset': -13},
                {'name': 'AIN1', 'gain': 2, "make_zero": True}],
      streamer=True,  # Mandatory to put the InOut in streamer mode

      # Sticking to default for the other arguments
  )

  # This Grapher Block plots an overview of the acquired data it receives from
  # the IOBlock, but not all the data
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'AIN0'), ('t(s)', 'AIN1'),

      # Sticking to default for the other arguments
  )

  # This HDFRecorder Block saves the data acquired by the IOBlock to a .hdf5
  # file
  rec = crappy.blocks.HDFRecorder(
      "data.h5",  # The file in which the data will be recorded
      atom=tables.Float64Atom(),  # The expected data type to be saved

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(labjack, graph,
              modifier=crappy.modifier.Demux(labels=('AIN0', 'AIN1')))
  crappy.link(labjack, rec)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
