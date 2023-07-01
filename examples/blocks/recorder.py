# coding: utf-8

"""
This example demonstrates the use of the Recorder Block. It does not require
any hardware to run, but necessitates the psutil module to be installed.

The Recorder Block saves the data it receives to a .csv file created at the
desired location. It can only record data from one Block, so multiple Recorders
must be used for recording data from multiple Blocks.

Here, a Recorder Block is used for saving the data recorded by an IOBlock
driving a FakeInOut that measures the RAM usage of the computer.

After starting the script, a new file is created at demo_recorder/data.csv.
Nothing visual should happen. The test should then be stopped by hitting
CTRL+C. After stopping it, you can check that the data was recorded by opening
the created .csv file. Try to open and close memory-intensive applications
(like web browsers) during the test to see important RAM variations in the
data.
"""

import crappy

if __name__ == '__main__':

  # This IOBlock acquires the data for the Recorder Block to save
  # It acquires data from the FakeInOut InOut, that reads the current RAM usage
  # of the computer
  mem = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut to read data from
      labels=('t(s)', 'memory(%)'),  # The names of the labels to send to
      # downstream Blocks
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Recorder Block records the data it receives from the IOBlock
  # The data is recorded as text in a newly created .csv file
  rec = crappy.blocks.Recorder(
      'demo_recorder/data.csv',  # The path to the file where the data will be
      # saved. It does not need to already exist
      labels=('t(s)', 'memory(%)'),  # The names of the labels to record
      freq=20,  # No need to run at a higher frequency than the label to record

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(mem, rec)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
