# coding: utf-8

"""
This example demonstrates the use of the HDF5Recorder Block. IT does not
require any hardware to run, but necessitates the tables and psutil Python
modules to be installed.

This Block takes a stream input from an IOBlock and records it in a .hdf5 file.
It can only save data from streams, not from InOut objects in normal operating
mode.

Here, the HDF5Recorder records data sent by an IOBlock driving a FakeInOut. By
default, it records it to the newly created demo_hdf5_recorder folder in the
data.hdf5 file.

After starting this script, nothing more happens but data is being recorded to
the destination file. Stop the test by hitting CTRL+C after a few seconds, and
notice how the data has been written to the destination file. For reading it,
you'll need to load the data, for example using the h5py Python module.
"""

import crappy

if __name__ == '__main__':

  # This IOBlock acquires the data to record, using the FakeInOut InOut object
  # It is in streamer mode to send data as a stream to the HDF5Recorder
  streamer = crappy.blocks.IOBlock(
      'FakeInout',  # The name of the InOut object to drive
      labels=('t(s)', 'stream'),  # The labels carrying the time and the stream
      # information
      streamer=True,  # Mandatory to use the streamer mode of the InOut
      freq=100,  # Quite high frequency to demonstrate the capacity of the
      # recorder

      # Sticking to default for the other arguments
  )

  # This HDF5Recorder Block saves the data it receives from the streaming
  # IOBlock to a .hdf5 file located in the newly created demo_hdf5_recorder
  # folder
  recorder = crappy.blocks.HDFRecorder(
      'demo_hdf5_recorder/data.hdf5',  # The file where to save the data
      atom='float64',  # The expected data type to record
      label='stream',  # The label carrying the stream data
      freq=200,  # Using a high frequency to make sure the recording goes
      # smoothly

      # Sticking to default for the other arguments
  )

  # Linking the Blocks together so that each one sends and received the correct
  # information
  crappy.link(streamer, recorder)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
