# coding: utf-8

"""
This example demonstrates the use of the Camera Block, for the basic use case
of just recording the acquired images. It does not require any hardware to run.

It acquires images on a fake camera, and records part of them at the given
location. Before the test starts, it also lets the user adjust some settings on
the camera in a configuration window.

After starting this script, you can play with the parameters in the
configuration window. Once you're done, close the configuration window. Nothing
should happen, except images will start being recorded at the given location.
Stop the test after a few seconds by hitting CTRL+C, and check the destination
folder to see the recorded images.
"""

import crappy

if __name__ == '__main__':

  # The Block in charge of acquiring the images and recording them
  # It also displays a configuration windows before the test starts, in which
  # the user can tune a few parameters of the Camera
  # Here, a fake camera is used so that no hardware is required
  cam = crappy.blocks.Camera(
      'FakeCamera',  # Using the FakeCamera camera so that no hardware is
      # required
      config=True,  # Before the test starts, displays a configuration window
      # for configuring the camera
      display_images=False,  # Here, we don't want the images displayed
      save_images=True,  # The acquired images should be recorded during this
      # test
      img_extension='tiff',  # The images should be saved as .tiff files
      save_folder='demo_record_images',  # The images will be saved in this
      # folder, whose path can be relative or absolute
      save_period=10,  # Only one out of 10 images will be saved, to avoid
      # bothering you with tons of images
      save_backend=None,  # The first available backend will be used
      freq=40,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
