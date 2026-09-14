# coding: utf-8

"""
This example demonstrates the simplest use of the VisionBlock pipeline: images
are acquired by a CameraSource Block and displayed by an independent
ImageDisplayer Block. It does not require any hardware, but necessitates Pillow
and either the opencv-python or matplotlib module to be installed.

Unlike the Camera Block, CameraSource only acquires images. Displaying,
recording, and processing are performed by other Blocks that receive its images
through ImageLinks. Splitting these tasks makes it possible to choose a
different looping frequency for each task and to reuse the same image stream in
several consumers.

Here, CameraSource acquires images from FakeCamera and ImageDisplayer shows the
newest available frame. Before the test starts, CameraSource opens a generic
configuration window in which the FakeCamera resolution, acquisition rate, and
moving-line speed can be adjusted. A StopButton Block allows stopping the
script cleanly.

After starting this script, adjust the FakeCamera settings if desired and close
the configuration window. The display window then shows the acquired images.
Notice that the Camera source may acquire images faster than the displayer
updates: an ImageLink always exposes the newest frame, so a slower consumer can
skip intermediate frames without delaying acquisition. Click the stop button
to end the demo. CTRL+C also stops it, but is not the cleanest way to stop
Crappy.
"""

import crappy


if __name__ == '__main__':

  # This Block is only responsible for opening the Camera and acquiring images.
  # FakeCamera generates a moving greyscale pattern, so no camera is required.
  camera = crappy.blocks.vision.CameraSource(
      'FakeCamera',  # Name of the Camera implementation to open
      config=True,  # Show the generic Camera configuration window first
      freq=40,  # Maximum frequency at which CameraSource checks for frames

      # Sticking to defaults for the other arguments
  )

  # Display is handled by a separate VisionBlock. Its framerate can be lower
  # than the acquisition rate without slowing down CameraSource.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='VisionBlock basic display',  # Title of the display window
      framerate=20,  # Never refresh the window more than 20 times per second
      freq=40,  # Frequency at which this Block checks for a new image

      # Sticking to defaults for the other arguments
  )

  # This Block lets the user terminate every Block cleanly from a button.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Images cannot travel through a regular crappy.link. An ImageLink uses a
  # shared-memory image buffer and carries the corresponding metadata with it.
  crappy.img_link(camera, displayer)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
