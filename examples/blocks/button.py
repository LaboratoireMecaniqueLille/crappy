# coding: utf-8

"""
This example demonstrates the use of the Button Block. It does not require any
hardware to run.

This Block displays a button on which the user can click, keeps track of the
number of clicks, and sends it to downstream Blocks.

In this example, the number of clicks is simply displayed in the window of a
Dashboard Block. The Button and Dashboard windows may initially overlap, so
you may need to move one to uncover the other.

After starting this script, you should click on the button that appeared and
watch how the Dashboard reacts accordingly. You can click multiple times in
quick succession or leave a few seconds between clicks. To end this demo, click
on the stop button that appears.
"""

import crappy

if __name__ == '__main__':

  # The Button Block that displays the GUI and keeps track of the number of
  # times its button was clicked
  # It sends the number of clicks to the downstream Blocks
  button = crappy.blocks.Button(
      send_0=True,  # The value 0 will be sent before the first loop
      label='step',  # The number of clicks is sent over this label
      time_label='t(s)',  # The time information is carried by this label
      spam=False,  # The number of clicks is sent at each new click, not at
      # each loop
      freq=10,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # This Block displays the time value and the number of steps at the moment of
  # the last click on the button
  # It is here to demonstrate how the information is sent to downstream Blocks
  graph = crappy.blocks.Dashboard(
      ('t(s)', 'step'),  # Only the time and the number of steps are displayed
      nb_digits=2,  # Limit the precision to 2 decimal digits

      # Sticking to default for the other arguments
      )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(button, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
