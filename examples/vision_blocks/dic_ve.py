# coding: utf-8

"""
This example demonstrates how to build a video-extensometry pipeline with the
DICVEProcessor VisionBlock. It does not require any hardware, but necessitates
the opencv-python, matplotlib, and Pillow modules to be installed.

DICVEProcessor tracks between one and four textured patches and calculates
their displacement using digital image correlation. With at least two patches,
it also derives horizontal and vertical strain from the changing distance
between their centers. Unlike the older DICVE Block, DICVEProcessor only
processes images: acquisition, display, and recording can each be handled by a
different VisionBlock.

In this example, a Generator controls a synthetic horizontal strain applied to
Crappy's bundled speckle image. CameraSource publishes the resulting images to
both DICVEProcessor and ImageDisplayer. DICVEProcessor sends its measurements
to Grapher and its current patch boxes to ImageDisplayer through regular Links.
It also feeds the measured horizontal strain back to Generator so that the
loading path changes direction at the requested strain limits.

No patch coordinates are supplied in the script. Before the Blocks start,
DICVEProcessor therefore sends a required configuration request upstream.
CameraSource handles that request by opening the specialized DICVE
configuration window and returns the selected patches to the processor. This
exchange happens automatically when the two Blocks are connected by an
ImageLink and both ``config`` and ``allow_downstream_config`` are enabled on
CameraSource.

After starting the script, set a patch size of about 64 pixels in the
configuration window, apply the setting, and draw two to four patches over the
speckled sample by left-clicking and dragging. Close the configuration window
to start the test. The image window displays the moving patch overlays and the
graph compares time with the measured horizontal strain. The test ends after
three loading cycles; the stop button can end it earlier.
"""

import crappy


if __name__ == '__main__':

  # This textured image is distributed with Crappy and is well suited to image
  # correlation. ApplyStrainToImage will deform it according to CameraSource's
  # current synthetic Exx and Eyy settings.
  img = crappy.resources.speckle

  # Generator drives the artificial horizontal strain. Its conditions use the
  # Exx(%) measurement returned by DICVEProcessor as feedback.
  generator = crappy.blocks.Generator(
      ({'type': 'CyclicRamp',
        'speed1': 1,  # Stretching at 1 percent per second
        'speed2': -1,  # Relaxing at 1 percent per second
        'condition1': 'Exx(%)>20',  # Reverse after reaching 20 percent
        'condition2': 'Exx(%)<0',  # Start stretching again below zero
        'cycles': 3,  # End the test after three complete loading cycles
        'init_value': 0},),  # Initial strain command, in percent
      cmd_label='Exx(%)',  # CameraSource recognizes this synthetic strain
      freq=50,  # More frequent than image acquisition for smooth commands

      # Sticking to defaults for the other arguments
  )

  # CameraSource performs acquisition only. Giving image_generator replaces a
  # physical Camera with a generated one based on the bundled speckle image.
  camera = crappy.blocks.vision.CameraSource(
      '',  # The Camera name is ignored when image_generator is provided
      image_generator=crappy.tool.ApplyStrainToImage(img),
      config=True,  # Required because no DICVE patches are supplied below
      allow_downstream_config=True,  # Accept DICVE's specialized request
      freq=30,  # Target generated-image acquisition frequency

      # Sticking to defaults for the other arguments
  )

  # This VisionBlock receives images and performs only the DICVE calculation.
  # With patches=None, its upstream configuration request is mandatory.
  dic = crappy.blocks.vision.DICVEProcessor(
      patches=None,  # Select the textured patches interactively
      request_configuration=True,  # Ask CameraSource to run DICVEConfig
      method='Disflow',  # OpenCV dense optical-flow correlation
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)',
              'Disp(px)'),  # Labels for the calculated values
      freq=50,  # Check for newly published images at up to 50 Hz

      # Sticking to defaults for the other arguments
  )

  # The same source images are displayed independently from their processing.
  # Patch overlays arrive on a regular Link from DICVEProcessor below.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='DICVEProcessor patches',
      framerate=15,  # Display need not run as fast as image acquisition
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Grapher displays the horizontal strain measured by DICVEProcessor.
  graph = crappy.blocks.Grapher(
      ('t(s)', 'Exx(%)'),

      # Sticking to defaults for the other arguments
  )

  # This Block offers a clean way to interrupt the otherwise finite demo.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Generator commands use a regular Link because they are small data values.
  crappy.link(generator, camera)

  # The measured strain controls Generator's loading-path conditions and is
  # also plotted. A third regular Link carries the patch overlay to displayer.
  crappy.link(dic, generator)
  crappy.link(dic, graph)
  crappy.link(dic, displayer)

  # CameraSource publishes one shared image stream to two independent
  # consumers. Each ImageLink exposes the newest image and matching metadata.
  crappy.img_link(camera, dic)
  crappy.img_link(camera, displayer)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
