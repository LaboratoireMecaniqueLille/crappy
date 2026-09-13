# coding: utf-8

"""
This example demonstrates a video-extensometry pipeline using the
VideoExtensoProcessor VisionBlock. It does not require any hardware, but
necessitates the opencv-python, scikit-image, Pillow, and matplotlib modules to
be installed.

VideoExtensoProcessor tracks up to four contrasted spots and calculates strain
from the evolution of their center coordinates. It creates an independent
tracker Process for every selected spot. Unlike the older VideoExtenso Block,
the processor neither opens a Camera nor displays images itself. CameraSource
performs acquisition, while ImageDisplayer independently combines the images
with the spot-box overlays produced by VideoExtensoProcessor.

In this example, a Generator applies synthetic horizontal strain to the
``ve_markers`` sample image distributed with Crappy. CameraSource publishes the
deformed images to both VideoExtensoProcessor and ImageDisplayer. The processor
sends its measured strain to Grapher and back to Generator, and sends the
current spot boxes to ImageDisplayer through a regular Link.

VideoExtensoProcessor cannot receive initial spot coordinates directly, so it
always sends a required configuration request to its image source. CameraSource
must have both ``config`` and ``allow_downstream_config`` enabled to serve it.
The source then opens the specialized Video Extenso configuration window before
the test and returns the selected spots and detection threshold to the
processor. This setup exchange is routed automatically through the ImageLink
graph; it does not use a runtime regular Link.

After starting the script, use the configuration window to detect or manually
select all four dark spots. Adjust the threshold if necessary, apply the
settings, and close the window. The display then follows the spot boxes while
the graph shows the measured horizontal strain. The test ends after three
loading cycles; the stop button can end it earlier.
"""

import crappy


if __name__ == '__main__':

  # The bundled sample contains four dark markers on a light background.
  img = crappy.resources.ve_markers

  # Generator drives the artificial horizontal strain and uses the measured
  # VideoExtensoProcessor strain as feedback for changing ramp direction.
  generator = crappy.blocks.Generator(
      ({'type': 'CyclicRamp',
        'speed1': 1,  # Stretching at 1 percent per second
        'speed2': -1,  # Relaxing at 1 percent per second
        'condition1': 'Exx(%)>20',  # Reverse after reaching 20 percent
        'condition2': 'Exx(%)<0',  # Start stretching again below zero
        'cycles': 3,  # End the test after three complete loading cycles
        'init_value': 0},),  # Initial strain command, in percent
      cmd_label='Exx(%)',
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # CameraSource deforms the bundled image according to received Exx(%) and
  # publishes it. Configuration must be enabled for the downstream request.
  camera = crappy.blocks.vision.CameraSource(
      '',  # The Camera name is ignored when image_generator is provided
      image_generator=crappy.tool.ApplyStrainToImage(img),
      config=True,  # Required by VideoExtensoProcessor's configuration
      allow_downstream_config=True,  # Serve the specialized downstream window
      freq=30,

      # Sticking to defaults for the other arguments
  )

  # This processor asks its source to select the initial spots and threshold.
  extenso = crappy.blocks.vision.VideoExtensoProcessor(
      white_spots=False,  # Track dark spots over the light sample background
      num_spots=4,  # Require all four markers in the bundled image
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)'),
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # The displayer receives raw images from CameraSource and dynamic spot-box
  # overlays from VideoExtensoProcessor over two different kinds of link.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='VideoExtensoProcessor spots',
      framerate=15,
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Grapher displays the horizontal strain calculated from the spot positions.
  graph = crappy.blocks.Grapher(
      ('t(s)', 'Exx(%)'),

      # Sticking to defaults for the other arguments
  )

  # This Block offers a clean way to interrupt the otherwise finite demo.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Regular Links carry strain commands, measurements, and overlay objects.
  crappy.link(generator, camera)
  crappy.link(extenso, generator)
  crappy.link(extenso, graph)
  crappy.link(extenso, displayer)

  # ImageLinks carry the same generated frames to two independent consumers.
  crappy.img_link(camera, extenso)
  crappy.img_link(camera, displayer)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
