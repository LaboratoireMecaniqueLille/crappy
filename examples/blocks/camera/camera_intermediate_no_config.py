# coding: utf-8

"""
This example demonstrates the use of the Camera Block, for the use case of
displaying the acquired images but with the configuration window disabled. It
does not require any hardware to run, but necessitates the opencv-python module
to be installed.

It acquires images on a fake camera, and displays them in a small visualization
window. The difference with the basic examples is that the configuration
window is disabled. It means that the arguments to pass to the Camera must be
given as arguments of the Camera Block, and two arguments become mandatory.

After starting this script, the acquired images start being displayed right
away. This demo never ends, and must be stopped by hitting CTRL+C. You can
restart the script with different values for the parameters of the FakeCamera,
and see how it's reflected in the acquired images.
"""

import crappy

if __name__ == '__main__':

  # The width and height of the images to display
  width = 1080
  height = 720

  # The Block in charge of acquiring the images and displaying them
  # Here, a fake camera is used so that no hardware is required
  cam = crappy.blocks.Camera(
      'FakeCamera',  # Using the FakeCamera camera so that no hardware is
      # required
      config=False,  # No configuration window is displayed before the test
      # starts !
      display_images=True,  # During the test, the acquired images are
      # displayed in a dedicated window
      displayer_framerate=30,  # The maximum framerate for displaying the
      # images
      save_images=False,  # Here, we don't want the images to be recorded
      freq=40,  # Lowering the default frequency because it's just a demo

      # These arguments are passed to the FakeCamera and automatically set
      speed=10,
      fps=30,
      width=width,
      height=height,

      # These arguments are mandatory as they cannot be retrieved from the
      # configuration window before the test starts
      # They depend on the camera that is used
      img_shape=(height, width),
      img_dtype='uint8',

      # Sticking to default for the other arguments
      )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
