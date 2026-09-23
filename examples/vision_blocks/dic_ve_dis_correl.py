# coding: utf-8

"""
This example demonstrates a complex VisionBlock graph in which DICVE and DIS
correlation analyze the same images at the same time. It does not require any
hardware, but necessitates the opencv-python and matplotlib modules to be
installed.

One CameraSource applies synthetic horizontal strain to Crappy's bundled
speckle image. Its image stream fans out to four independent consumers:
DICVEProcessor tracks four smaller textured patches, DISCorrelProcessor
calculates dense optical flow over one central patch, ImageDisplayer provides
live monitoring, and ImageRecorder periodically saves the original images.
Although four ImageLinks are present, CameraSource owns one shared-memory image
buffer. Every consumer copies the newest coherent image and metadata when it is
ready, without serializing the array through a Pipe or forcing the consumers to
run at the same rate.

The two processors use different algorithms and publish distinct labels, so
their strain measurements can be plotted together. They also each send an
``'overlay'`` value to the same ImageDisplayer through separate regular Links.
ImageDisplayer retains the latest overlay from every incoming Link and draws
both the four DICVE boxes and the DISCorrel box over the shared source image.
This illustrates why image data and ordinary result data use separate graphs.

All patch coordinates are provided in the script. Camera configuration is
disabled, the source image format is declared explicitly, and both processors
have ``request_configuration=False``. The entire graph can therefore prepare
without opening configuration windows. The only windows shown during the test
are the live image, result graph, and stop button.

After starting the script, compare the commanded strain with the DICVE and
DISCorrel estimates in the graph, and observe both kinds of overlay in the
image window. One image in ten is saved as a NumPy array, together with its
metadata, under ``demo_combined_vision_images``. Existing recordings are not
overwritten; a numbered folder is selected when necessary. The Generator ends
the test after three cycles, and the stop button can end it earlier.
"""

import crappy


if __name__ == '__main__':

  # Both correlation methods analyze deformations of this same bundled image.
  img = crappy.resources.speckle

  # These coordinates are ordered as (y, x, height, width). DICVE tracks four
  # distributed regions, whereas DISCorrel analyzes one larger central region.
  dic_patches = ((100, 224, 64, 64),
                 (224, 348, 64, 64),
                 (348, 224, 64, 64),
                 (224, 100, 64, 64))
  dis_patch = (193, 193, 128, 128)

  # Generator produces the strain applied by CameraSource. The self-Link below
  # returns its own Exx(%) command for evaluating the path conditions, leaving
  # the two measured strains available for comparison rather than control.
  generator = crappy.blocks.Generator(
      ({'type': 'CyclicRamp',
        'speed1': 1,  # Stretching at 1 percent per second
        'speed2': -1,  # Relaxing at 1 percent per second
        'condition1': 'Exx(%)>20',  # Reverse at 20 percent commanded strain
        'condition2': 'Exx(%)<0',  # Begin the next loading cycle below zero
        'cycles': 3,
        'init_value': 0},),
      cmd_label='Exx(%)',  # Label recognized by the generated Camera source
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # The image source is configured entirely by arguments. Output shape and
  # dtype are mandatory because no configuration window will inspect a frame.
  camera = crappy.blocks.vision.CameraSource(
      '',  # The Camera name is ignored when image_generator is provided
      image_generator=crappy.tool.ApplyStrainToImage(img),
      config=False,
      allow_downstream_config=False,
      img_shape=img.shape,
      img_dtype=str(img.dtype),
      freq=30,

      # Sticking to defaults for the other arguments
  )

  # DICVE calculates strain from relative motion of four textured patches.
  # Distinct labels prevent its values from being confused with DIS results.
  dic = crappy.blocks.vision.DICVEProcessor(
      patches=dic_patches,
      request_configuration=False,
      method='Disflow',
      labels=('t(s)', 'meta', 'DIC Coord(px)', 'DIC Eyy(%)',
              'DIC Exx(%)', 'DIC Disp(px)'),
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # DISCorrel projects dense optical flow over the central patch onto the two
  # normal-strain fields. Its result labels are unique to this processor.
  correl = crappy.blocks.vision.DISCorrelProcessor(
      patch=dis_patch,
      fields=('exx', 'eyy'),
      labels=('t(s)', 'meta', 'DIS Exx(%)', 'DIS Eyy(%)'),
      request_configuration=False,
      follow=True,  # Keep the patch centered during the large deformation
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # This single window receives source images through an ImageLink and the two
  # processors' independently updated overlays through regular Links.
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='DICVE and DISCorrel on one sample',
      framerate=15,
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Recording is independent from acquisition, processing, and display. The
  # NumPy backend preserves the exact greyscale arrays without optional codecs.
  recorder = crappy.blocks.vision.ImageRecorder(
      save_folder='demo_combined_vision_images',
      save_period=10,  # Save at most one out of ten published source frames
      save_backend='npy',
      freq=50,

      # Sticking to defaults for the other arguments
  )

  # Plot the applied strain and both measurements in the same axes. Grapher can
  # receive matching label pairs from several independent regular Links.
  graph = crappy.blocks.Grapher(
      ('t(s)', 'Exx(%)'),
      ('t(s)', 'DIC Exx(%)'),
      ('t(s)', 'DIS Exx(%)'),

      # Sticking to defaults for the other arguments
  )

  # This Block provides a clean manual exit before the three cycles complete.
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # The command is sent to CameraSource, Grapher, and back to Generator itself.
  crappy.link(generator, camera)
  crappy.link(generator, graph)
  crappy.link(generator, generator)

  # Processor outputs carry measurements and overlays through regular Links.
  crappy.link(dic, graph)
  crappy.link(dic, displayer)
  crappy.link(correl, graph)
  crappy.link(correl, displayer)

  # Four ImageLinks fan the same source buffer out to independent consumers.
  # Explicit names make the topology and receive-side diagnostics clearer.
  crappy.img_link(camera, dic, name='camera-to-dic')
  crappy.img_link(camera, correl, name='camera-to-dis')
  crappy.img_link(camera, displayer, name='camera-to-display')
  crappy.img_link(camera, recorder, name='camera-to-recorder')

  # Mandatory line for starting the test; this call is blocking.
  crappy.start()
