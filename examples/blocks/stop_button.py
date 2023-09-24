# coding: utf-8

"""
This example demonstrates the use of the StopButton Block. It does not require
any hardware nor specific Python module to run.

This Block simply allows to stop a test when clicking on a button in a GUI. It
constitutes one of the clean ways to stop a Crappy script.

Here, only one StopButton Block is instantiated, waiting to be clicked.

After starting this script, just click on the button that appeared and it will
stop the test. You can also stop this script by hitting CTRL+C, but it is not a
proper way to end a script in Crappy.
"""

import crappy

if __name__ == '__main__':

  # This StopButton Block will stop the test once it is clicked
  button = crappy.blocks.StopButton(
      # No specific argument to set here
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
