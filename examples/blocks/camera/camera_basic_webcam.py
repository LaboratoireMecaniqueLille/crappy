# coding: utf-8

"""
This example demonstrates the use of the Camera Block, for the basic use case
of just displaying the acquired images. It requires a camera able to interface
with OpenCV to be connected, typically an integrated or external webcam can do.

It acquires images from the webcam, and displays them in a small visualization
window. Before the test starts, it also lets the user adjust some settings on
the camera in a configuration window.

After starting this script, a very basic configuration window appears. Once
you're done playing with it, close it and watch the displayer broadcast the
acquired images. This demo never ends, and must be stopped by hitting CTRL+C.
"""

import crappy

if __name__ == '__main__':

  # The Block in charge of acquiring the images and displaying them
  # It also displays a configuration windows before the test starts, in which
  # the user can tune a few parameters of the Webcam
  # Here, the very basic Webcam Camera is used
  cam = crappy.blocks.Camera(
      'Webcam',  # Using the Webcam to acquire the images
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
