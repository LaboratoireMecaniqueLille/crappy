# coding: utf-8

"""
This example demonstrates a non-interactive image-correlation pipeline using
the DISCorrelProcessor VisionBlock. It does not require any hardware, but
necessitates the opencv-python and matplotlib modules to be installed.

DISCorrelProcessor calculates dense optical flow over one rectangular image
patch, then projects that flow onto one or more displacement or deformation
fields. Crappy can generate common fields from names such as ``'x'``, ``'y'``,
``'exx'``, and ``'eyy'``. A user can also supply any finite NumPy vector field
with the shape ``(patch_height, patch_width, 2)``. This example mixes both
kinds: Crappy generates the translation fields, while equivalent horizontal
and vertical strain fields are built explicitly below.

The patch coordinates are also supplied directly. Consequently neither
CameraSource nor DISCorrelProcessor needs an interactive configuration window.
When CameraSource configuration is disabled, its output image shape and dtype
must be stated in advance so that the shared-memory buffer can be allocated
during preparation. Likewise, ``request_configuration=False`` tells the
processor not to send even an optional configuration request upstream.

A Generator applies cyclic horizontal strain to Crappy's bundled speckle image.
CameraSource publishes each generated image to DISCorrelProcessor and an
ImageDisplayer. The processor sends translations, projected strains, and a
correlation residual as regular data. It also sends the selected patch as an
overlay for the displayer. The measured horizontal strain controls the
Generator's loading path and is plotted together with the residual.

Start the script and the test begins immediately, without configuration. The
display window shows the predefined correlation patch, and the graph follows
the measured strain and residual. The test ends after three loading cycles; the
stop button can end it earlier. Try replacing the custom strain arrays with the
strings ``'exx'`` and ``'eyy'`` to use Crappy's generated fields instead.
"""

import numpy as np

import crappy


if __name__ == '__main__':

  # Loading the textured sample distributed with Crappy. It is a greyscale
  # uint8 array with shape (512, 512).
  img = crappy.resources.speckle

  # Defining the fixed region on which dense optical flow will be calculated.
  # Patch coordinates are ordered as (y, x, height, width).
  patch = (193, 193, 128, 128)
  _, _, patch_height, patch_width = patch

  # Building two custom vector fields of shape (height, width, 2). They are
  # equivalent to Crappy's generated 'exx' and 'eyy' fields and demonstrate how
  # application-specific projection fields can be supplied as NumPy arrays.
  exx = np.stack(
      (np.tile(np.linspace(-patch_width / 200, patch_width / 200,
                           patch_width, dtype=np.float32),
               (patch_height, 1)),
       np.zeros((patch_height, patch_width), dtype=np.float32)),
      axis=2)
  eyy = np.stack(
      (np.zeros((patch_height, patch_width), dtype=np.float32),
       np.tile(np.linspace(-patch_height / 200, patch_height / 200,
                           patch_height, dtype=np.float32)[:, np.newaxis],
               (1, patch_width))),
      axis=2)

  # Generator drives the artificial strain. The calculated Exx(%) value from
  # DISCorrelProcessor is fed back to evaluate these transition conditions.
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

  # With configuration disabled, the output buffer's format has to be known
  # from construction. image_generator makes a physical Camera unnecessary.
  camera = crappy.blocks.vision.CameraSource(
      '',  # The Camera name is ignored when image_generator is provided
      image_generator=crappy.tool.ApplyStrainToImage(img),
      config=False,  # Start without opening a Camera configuration window
      allow_downstream_config=False,  # Reject optional processor requests
      img_shape=img.shape,  # Shape required for shared-memory allocation
      img_dtype=str(img.dtype),  # Dtype required for shared-memory allocation
      freq=30,

      # Sticking to defaults for the other arguments
  )

  # This processor mixes generated translation fields and custom strain fields.
  # Setting residual=True appends a 'res' value to the labels automatically.
  correl = crappy.blocks.vision.DISCorrelProcessor(
      patch=patch,  # Predefined (y, x, height, width) correlation region
      fields=('x', 'y', exx, eyy),
      labels=('t(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)'),
      request_configuration=False,  # The supplied patch is already complete
      residual=True,  # Also report the mean absolute correlation residual
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Images and the latest patch overlay are combined only in the displayer.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='DISCorrelProcessor patch',
      framerate=15,
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Plot the custom horizontal strain projection and correlation residual.
  graph = crappy.blocks.Grapher(
      ('t(s)', 'Exx(%)'),
      ('t(s)', 'res'),

      # Sticking to defaults for the other arguments
  )

  # This Block offers a clean way to interrupt the otherwise finite demo.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Small command and result dictionaries use regular Links.
  crappy.link(generator, camera)
  crappy.link(correl, generator)
  crappy.link(correl, graph)
  crappy.link(correl, displayer)  # Supplies the patch under 'overlay'

  # Image arrays and matching metadata use shared-memory ImageLinks.
  crappy.img_link(camera, correl)
  crappy.img_link(camera, displayer)

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
