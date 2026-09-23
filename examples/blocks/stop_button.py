# coding: utf-8

"""
This example demonstrates the use of the StopButton Block. It requires neither
hardware nor any specific Python modules.

This Block provides a GUI button that cleanly stops a test when clicked.

Here, only one StopButton Block is instantiated, waiting to be clicked.

After starting this script, click the button to stop the test.
"""

import crappy

if __name__ == '__main__':

  # This StopButton Block will stop the test once it is clicked
  button = crappy.blocks.StopButton(
      # No specific argument to set here
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
