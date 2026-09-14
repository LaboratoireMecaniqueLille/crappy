# coding: utf-8

"""
This example demonstrates how to record images with independent VisionBlocks.
It does not require any hardware or optional image-writing backend to run, but
Pillow is required by the interactive Camera configuration window.

A CameraSource Block acquires images from FakeCamera and publishes them through
an ImageLink. An ImageRecorder Block receives the image stream and saves one
frame out of ten with NumPy. It also creates a metadata.csv file containing the
metadata associated with every saved frame. Acquisition and disk writing run in
different processes, so a slow storage device does not directly block the
CameraSource. As with every ImageLink consumer, the recorder handles the newest
available frame and may skip images if it cannot keep up.

The example also illustrates the difference between Crappy's two link types.
The images and their metadata travel from CameraSource to ImageRecorder through
an ImageLink. Once an image is actually saved, ImageRecorder sends a small
notification dictionary through a regular Link, and LinkReader prints it in the
console. The notification contains the saved image's timestamp, index, and full
metadata, but not the image array itself.

After starting this script, adjust the FakeCamera settings if desired and close
the configuration window. Let the test run for a few seconds, then click the
stop button. The images and metadata are written to the
demo_vision_record_images folder. If that folder already contains a Crappy
recording, ImageRecorder selects a new folder with a numeric suffix instead of
overwriting it. CTRL+C also stops the demo, but is not the cleanest way to stop
Crappy.
"""

import crappy


if __name__ == '__main__':

  # CameraSource only acquires and publishes images. FakeCamera makes the
  # example runnable without a physical camera.
  camera = crappy.blocks.vision.CameraSource(
      'FakeCamera',  # Name of the Camera implementation to open
      config=True,  # Let the user configure FakeCamera before recording
      freq=40,  # Maximum frequency at which CameraSource checks for frames

      # Sticking to defaults for the other arguments
  )

  # ImageRecorder receives one image stream and writes selected frames to disk.
  # The NumPy backend is always available and preserves the array exactly.
  recorder = crappy.blocks.vision.ImageRecorder(
      save_folder='demo_vision_record_images',  # Recording destination
      save_period=10,  # Save at most one image for every ten source images
      save_backend='npy',  # Write arrays as .npy without an optional backend
      freq=40,  # Frequency at which the recorder checks for a new frame

      # Sticking to defaults for the other arguments
  )

  # The recorder emits a regular message only after an image has been saved.
  # LinkReader makes these notifications visible without opening another GUI.
  reader = crappy.blocks.LinkReader(
      name='Saved image',  # Prefix used for messages printed in the console
      freq=20,  # More than enough for the reduced recording rate

      # Sticking to defaults for the other arguments
  )

  # This Block lets the user terminate every Block cleanly from a button.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # The full images and their metadata travel in shared memory through an
  # ImageLink. Only a lightweight saved-frame notification uses a regular Link.
  crappy.img_link(camera, recorder)
  crappy.link(recorder, reader)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
