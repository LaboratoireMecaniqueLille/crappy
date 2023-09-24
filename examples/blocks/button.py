# coding: utf-8

"""
This example demonstrates the use of the Button Block. It does not require any
hardware to run.

This Block displays a button on which the user can click, keeps track of the
number of clicks, and sends it to downstream Blocks.

In this example, the number of clicks is simply displayed on the window of a
Dashboard Block. The windows of the Button and the Dashboard might first be
overlapping, you may need to move one to uncover the other. Note that in
addition, A StopButton Block allows stopping the script properly without using
CTRL+C by clicking on a button.

After starting this script, you should click on the button that appeared and
watch how the Dashboard reacts accordingly. You can try to click multiple times
at once, or to leave a few seconds between you clicks. To end this demo, click
on the stop button that appears. You can also hit CTRL+C, but it is not a clean
way to stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # The Button Block that displays the GUI and keeps track of the number of
  # times its button was clicked
  # It sends the number of clicks to the downstream Blocks
  button = crappy.blocks.Button(
      send_0=True,  # The value 0 will be sent until the button is clicked
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
