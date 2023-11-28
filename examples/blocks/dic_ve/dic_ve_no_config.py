# coding: utf-8

"""
This example demonstrates the use of the DICVE Block in the case when the
patches are manually entered by the user. It does not require any hardware to
run, but necessitates the opencv-python and matplotlib modules to be installed.

It is the exact same script as dic_ve_basic.py, except the patches are manually
provided and the configuration window is disabled. Refer to the other script
for more information.
"""

import crappy

if __name__ == '__main__':

  # Loading the example image of a speckle used for performing the image
  # correlation. This image is distributed with Crappy
  img = crappy.resources.speckle

  # The Generator Block that drives the fake strain on the image
  # It applies a cyclic strain that makes the image stretch in the x direction
  gen = crappy.blocks.Generator(
      # Using a CyclicRamp Path to generate cyclic linear stretching
      ({'type': 'CyclicRamp',
        'speed1': 1,  # Stretching at 1%/s
        'speed2': -1,  # Relaxing at 1%/s
        'condition1': 'Exx(%)>20',  # Stretching until 20% strain
        'condition2': 'Exx(%)<0',  # Relaxing until 0% strain
        'cycles': 3,  # The test stops after 3 cycles
        'init_value': 0},),  # Mandatory to give as it's the first Path
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='Exx(%)',  # The generated signal corresponds to a strain

      # Sticking to default for the other arguments
  )

  # This DICVE Block calculates the strain of the image by tracking the
  # displacement of the patches
  # This Block is actually also the one that generates the fake strain on the
  # image, but that wouldn't be the case in real life
  # It takes the target strain as an input, and outputs both the computed
  # strain and the positions of the tracked patches
  dic_ve = crappy.blocks.DICVE(
      '',  # The name of Camera to open is ignored because image_generator is
      # given
      config=False,  # Not displaying the configuration window before starting
      # the test, the patches must be provided
      display_images=True,  # The displayer window will allow to follow the
      # patches on the speckle image
      freq=50,  # Lowering the default frequency because it's just a demo
      save_images=False,  # We don't want images to be recorded in this demo
      image_generator=crappy.tool.ApplyStrainToImage(img),  # This argument
      # makes the Block generate fake strain on the given image, only useful
      # for demos
      patches=[(100, 224, 64, 64), (224, 348, 64, 64),
               (348, 224, 64, 64), (224, 100, 64, 64)],  # Providing here the
      # patches to track as they cannot be selected in the configuration window
      # The labels for sending the calculated strain to downstream Blocks
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'Disp(px)'),
      method='Disflow',  # The default image correlation method

      # Arguments to provide because the configuration window is disabled
      img_dtype='uint8',
      img_shape=(512, 512),

      # Sticking to default for the other arguments
  )

  # This Grapher displays the extension as computed by the DICVE Block
  graph = crappy.blocks.Grapher(('t(s)', 'Exx(%)'))

  # Linking the Blocks together so that each one sends and received the correct
  # information
  # The Generator drives the DICVE, but also takes decision based on its
  # feedback
  crappy.link(gen, dic_ve)
  crappy.link(dic_ve, gen)
  crappy.link(dic_ve, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
