# coding: utf-8

"""
This example demonstrates software-triggered acquisition with a CameraSource
Block. It does not require any hardware, but necessitates Pillow and either the
opencv-python or matplotlib module to be installed.

CameraSource can wait for a label received through a regular Link before asking
its Camera for an image. Here, a Button Block sends the trigger label whenever
the user clicks it. The value carried under that label is irrelevant; receiving
the label is the event that authorizes one acquisition attempt. Each acquired
image is then published through an ImageLink and shown by an independent
ImageDisplayer Block.

This example intentionally uses both kinds of Crappy link. The regular Link
between Button and CameraSource carries a small command dictionary through a
Pipe. The ImageLink between CameraSource and ImageDisplayer carries image data
and matching metadata through shared memory. Software triggering is convenient
for manual and low-rate acquisition, but it is not deterministic enough for
precision triggering and should generally stay below approximately 10 Hz.

After starting this script, adjust the FakeCamera settings if desired and close
the configuration window. Click the trigger button to acquire and display a new
frame; the display remains on the last acquired image between clicks. Click the
separate stop button to end the demo cleanly.
"""

import crappy


if __name__ == '__main__':

  # Button sends a new value only when it is clicked. CameraSource only checks
  # whether the 'trigger' label is present, not which value it contains.
  trigger = crappy.blocks.Button(
      send_0=False,  # Do not acquire an image automatically at test startup
      label='trigger',  # Label CameraSource waits for before acquiring
      time_label='t(s)',  # Label carrying the click timestamp
      spam=False,  # Send only on clicks, rather than at every Button loop
      freq=10,  # Appropriate rate for this manual software trigger

      # Sticking to defaults for the other arguments
  )

  # CameraSource opens FakeCamera and waits for a regular message containing
  # 'trigger' before every acquisition attempt.
  camera = crappy.blocks.vision.CameraSource(
      'FakeCamera',  # Name of the Camera implementation to open
      software_trig_label='trigger',  # Label enabling an acquisition
      config=True,  # Configure FakeCamera before triggering becomes active
      freq=40,  # Frequency at which trigger messages are checked

      # Sticking to defaults for the other arguments
  )

  # The images acquired after clicks are displayed by another VisionBlock.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='Software-triggered images',  # Title of the display window
      framerate=20,  # Upper limit; actual updates follow manual trigger clicks
      freq=40,  # Frequency at which the ImageLink is checked for a new frame

      # Sticking to defaults for the other arguments
  )

  # This is separate from the trigger button and cleanly ends the entire test.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Trigger commands are small dictionaries and therefore use a regular Link.
  crappy.link(trigger, camera)

  # Acquired arrays and their metadata require an ImageLink.
  crappy.img_link(camera, displayer)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
