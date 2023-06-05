# coding: utf-8

"""
This example demonstrates the use of the Camera Block, for the basic use case
of just displaying the acquired images. It does not require any hardware to
run.

It acquires images on a fake camera, and displays them in a small visualization
window. Before the test starts, it also lets the user adjust some settings on
the camera in a configuration window.

After starting this script, you can play with the parameters in the
configuration window. Once you're done, close the configuration and watch the
displayer broadcast the images with the chosen settings. You can run this
script multiple times and set new parameters. This demo never ends, and must be
stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # The Block in charge of acquiring the images and displaying them
  # It also displays a configuration windows before the test starts, in which
  # the user can tune a few parameters of the Camera
  # Here, a fake camera is used so that no hardware is required
  cam = crappy.blocks.Camera(
      'FakeCamera',  # Using the FakeCamera camera so that no hardware is
      # required
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

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
