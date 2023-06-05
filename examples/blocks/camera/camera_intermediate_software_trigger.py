# coding: utf-8

"""
This example demonstrates the use of the Camera Block, for the use case of
triggering the image acquisition using a software trigger. The acquired images
are then displayed. It does not require any hardware to run.

It acquires images on a fake camera, and displays them in a small visualization
window. Before the test starts, it also lets the user adjust some settings on
the camera in a configuration window. The difference with the basic display
example is that here a Button Block lets the user decide when to acquire the
images. The trigger is only active once the test starts, so the configuration
window runs normally.

After starting this script, you can play with the parameters in the
configuration window. Once you're done, close the configuration window. Then,
click on the button to trigger an image acquisition and watch the acquired
images be broadcast in the displayer window. This demo never ends, and must be
stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # The Button Block that displays the clickable button and sends a signal when
  # it is clicked
  button = crappy.blocks.Button(
      send_0=False,  # No value is sent before the user first clicks
      label='trigger',  # The number of clicks is sent over this label
      time_label='t(s)',  # The time information is carried by this label
      spam=False,  # The number of clicks is sent at each new click, not at
      # each loop
      freq=10,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # The Block in charge of acquiring the images and displaying them
  # It will only trigger an acquisition when receiving a signal over the
  # specified label
  # It also displays a configuration windows before the test starts, in which
  # the user can tune a few parameters of the Camera
  # Here, a fake camera is used so that no hardware is required
  cam = crappy.blocks.Camera(
      'FakeCamera',  # Using the FakeCamera camera so that no hardware is
      # required
      software_trig_label='trigger',  # A frame will be acquired each time a
      # value is received over this label. The received value does not matter
      config=True,  # Before the test starts, displays a configuration window
      # for configuring the camera
      display_images=True,  # During the test, the acquired images are
      # displayed in a dedicated window
      displayer_framerate=30,  # The maximum framerate for displaying the
      # images
      save_images=False,  # Here, we don't want the images to be recorded
      freq=40,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # Linking the Block so that the information is correctly sent and received
  crappy.link(button, cam)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
