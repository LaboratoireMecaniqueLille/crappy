# coding: utf-8

"""
This example demonstrates the use of the StopBlock. It does not require any
hardware nor specific Python module to run.

This Block checks if given criteria are met in the data it receives. If so, it
triggers an event that stops the entire Crappy script. It is one of the clean
ways to stop a Crappy script.

Here, a Button Block is linked to a StopBlock. If the click count of the Button
Block reaches 6, or if 10 seconds are elapsed, the StopBlock stops the script.

After starting this script, you can either click 6 times on the button that
appears, or wait for 10 seconds. In both cases, the script should then stop by
itself. You can also stop this script earlier than 10s by hitting CTRL+C (not
a proper way to end a script in Crappy).
"""

import crappy

if __name__ == '__main__':

  # This Button Block increments a counter at each click on a GUI button, and
  # sends the click count to the StopBlock
  button = crappy.blocks.Button(
      label='crit',  # The label carrying the number of clicks
      freq=30,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This StopBlock checks if the data from the Button Block satisfies any of
  # its stop criteria, in which case it stops the test
  stop = crappy.blocks.StopBlock(
      criteria=('crit > 5', 't(s)>10'),  # The stop criterion, spaces in the
      # strings do not matter
      display_freq=True,  # Just to show in the terminal that the Blocks are
      # not frozen

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(button, stop)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
