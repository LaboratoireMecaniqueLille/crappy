# coding: utf-8

"""
This example demonstrates how the Camera Block records acquired images and
sends a message to downstream Blocks each time a new image is saved. It does
not require any hardware to run, but necessitates the opencv-python,
scikit-image and Pillow modules to be installed.

It acquires images from a fake camera and records some of them at the given
location. Before the test starts, it also lets the user adjust some settings on
the camera in a configuration window. Because images are recorded, no image
processing is performed. When an output Link is defined, the Camera sends a
message to downstream Blocks each time it saves a new image. A Dashboard Block
receives this message and displays the most recent image-saving timestamp.

After starting this script, you can play with the parameters in the
configuration window. Once you're done, close the configuration window. Images
will start being recorded at the given location, and the timestamp of the most
recently saved image should be displayed by the Dashboard. Stop the test after
a few seconds by clicking on the stop button that appears, and check the
destination folder to see the recorded images.
"""

import crappy

if __name__ == '__main__':

  # The Block in charge of acquiring the images and recording them
  # It also displays a configuration window before the test starts, in which
  # the user can tune a few parameters of the Camera
  # Here, a fake camera is used so that no hardware is required
  # Because save_images is True, no image processing is performed, and an
  # output Link is defined, the timestamp and metadata are sent for each newly
  # saved image under the labels 't(s)' and 'meta'
  cam = crappy.blocks.Camera(
      'FakeCamera',  # Using the FakeCamera camera so that no hardware is
      # required
      config=True,  # Before the test starts, displays a configuration window
      # for configuring the camera
      display_images=False,  # Here, we don't want the images displayed
      save_images=True,  # The acquired images should be recorded during this
      # test
      img_extension='tiff',  # The images should be saved as .tiff files
      save_folder='demo_record_images_send',  # The images will be saved in
      # this folder, whose path can be relative or absolute
      save_period=10,  # Only one out of ten images will be saved, to avoid
      # bothering you with tons of images
      save_backend=None,  # The first available backend will be used
      freq=40,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
      )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # This Block displays the time value of the moments when an image is saved by
  # the Camera Block
  # It is here to demonstrate that the information is properly sent to
  # downstream Blocks
  dash = crappy.blocks.Dashboard(('t(s)',))

  # Linking the Blocks together so that the correct information is sent
  crappy.link(cam, dash)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
