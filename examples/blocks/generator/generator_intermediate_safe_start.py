# coding: utf-8

"""
This example demonstrates the use of the safe_start argument of the Generator
Block. It does not require any hardware to run.

The Generator Block outputs a signal following a provided path. Several paths
are available, each with a different behavior and different options. They can
be combined to form a custom global path.

Here, the safe_start argument of the Generator is set to True, so it waits for
a first value of 'control' to arrive before it outputs any data. Because
'control' is generated by a Button Block, the user decides when the Generator
starts generating data.

After starting this script, the Grapher remains empty as long as the user does
not click on the Button to generate a first value of 'control'. Then, the
signal starts to be generated and is displayed on the Grapher. This demo never
ends, and must be stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # This Generator outputs a ramp signal
  # It is used here to demonstrate the use of the safe_start argument
  gen = crappy.blocks.Generator(
      # Generating a simple ramp, that never ends because 'control' never drops
      # below zero. Because 'safe_start' is True, no signal is output before
      # a value is received for 'control'
      ({'type': 'Ramp',
        'speed': 1,
        'condition': 'control<0',  # The 'control' label is the condition here
        'init_value': 0},),
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='signal',  # The label carrying the value of the generated
      # signal
      path_index_label='index',  # This label carries the index of the current
      # path
      repeat=False,  # When reaching the end of the path, stop the test and do
      # not repeat the path forever
      spam=False,  # Only send a value if it's different from the last sent one
      end_delay=2,  # When the path is exhausted, wait for 2 seconds before
      # ending the test
      safe_start=True,  # Before sending the first value, making sure that at
      # least one value is received on the 'control' label

      # Sticking to default for the other arguments
      )

  # This Button generates the label used as a condition by the Generator Block
  # The Generator starts to output data as soon as this Button is clicked
  button = crappy.blocks.Button(
      send_0=False,  # Do not send the first 0 value when starting
      label='control',  # The label carrying the click information
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # This Grapher displays the signal it receives from the Generator
  graph = crappy.blocks.Grapher(('t(s)', 'signal'))

  # Linking the Block so that the information is correctly sent and received
  crappy.link(gen, graph)
  crappy.link(button, gen)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
