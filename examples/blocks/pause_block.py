# coding: utf-8

"""
This example demonstrates the use of the Pause Block. It does not require any
hardware, but requires the psutil Python module.

This Block pauses other Blocks during a test based on a given set of
conditions, then resumes them later. This can be useful when human intervention
is needed on a setup during a test, ensuring that no command is sent to the
hardware during that time.

Here, a Generator Block generates a signal on which a Pause Block
decides to pause or resume the other Blocks. The Generator is of course
configured to be insensitive to the pauses. In parallel, an IOBlock monitors
the current RAM usage and sends it to a LinkReader Block for display.

After starting this script, the values acquired by the IOBlock appear in the
console. After 8 seconds, they stop because the IOBlock is paused. After 12
seconds, it resumes and the values appear again. The same happens after 28
seconds, except that the pause never ends because a second pause condition is
satisfied when t > 30 seconds. To end this demo, click the stop button.
"""

import crappy

if __name__ == '__main__':

  # This Generator Block generates a cyclic ramp signal, and that is sent to
  # the Pause Block
  gen = crappy.blocks.Generator(
      # Generating a cyclic ramp signal, oscillating in a linear way between 0
      # and 10 with a period of 20 seconds
      ({'type': 'CyclicRamp', 'init_value': 0, 'cycles': 0,
        'condition1': 'delay=10', 'condition2': 'delay=10',
        'speed1': 1, 'speed2': -1},),
      cmd_label='value',  # The labels carrying the generated value
      freq=10,  # Setting a low frequency because we don't need more

      # Sticking to default for the other arguments
  )
  # Extremely important line, prevents the Generator from being paused
  # Otherwise, the signal checked by the Pause Block ceases to be generated and
  # the pause therefore never ends
  gen.pausable = False

  # This Block checks whether any pause criteria are met and, if so, pauses all
  # pausable Blocks
  pause = crappy.blocks.Pause(
      # The pause lasts as long as the "value" label is higher than 8, or when
      # the time reaches 30 seconds
      criteria=('value>8', 't(s)>30'),
      freq=20,  # Setting a low frequency because we don't need more

      # Sticking to default for the other arguments
  )

  # This IOBlock reads the current memory usage of the system, and sends it to
  # the LinkReader
  io = crappy.blocks.IOBlock(
      'FakeInOut',  # The name of the InOut object to drive
      labels=('t(s)', 'memory'),  # The names of the labels to output
      freq=5,  # Low frequency to avoid spamming the console
      display_freq=True,  # Display the looping frequency to show that there
      # are still loops, although no data is acquired

      # Sticking to default for the other arguments
  )

  # This LinkReader Block displays in the console the data it receives from the
  # IOBlock
  reader = crappy.blocks.LinkReader(
      name='Reader',  # A name for identifying the Block in the console
      freq=5,  # Useless to set a frequency higher than the labels to display

      # Sticking to default for the other arguments
  )
  # During the pause, no data is displayed because the IOBlock is on hold and
  # not because the LinkReader is paused
  reader.pausable = False

  # This Block allows the user to properly exit the script
  # By default, it is not affected by the pause mechanism
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, pause)
  crappy.link(io, reader)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
