# coding: utf-8

"""
This example demonstrates the use of the VideoExtenso Block. It does not
require any hardware to run, but necessitates the opencv-python, scikit-image,
Pillow and matplotlib modules to be installed.

This Block computes the strain on acquired images by tracking the displacement
of several spots. It outputs the computed strain as well as the position and
displacement of the spots.

In this example, a fake strain is generated on a static image of a sample with
spots drawn on it. The level of strain is controlled by a Generator Block, and
applied to the images by the VideoExtenso Block. This same VideoExtenso Block
then calculates the strain on the images, and outputs it to a Grapher Block for
display.

After starting this script, you have to select the spots to track in the
configuration window by left-clicking and dragging. Then, close the
configuration window and watch the strain be calculated in real time. This demo
normally ends automatically after 2 minutes. You can also hit CTRL+C to stop it
earlier, but it is not a clean way to stop Crappy.
"""

import crappy

if __name__ == '__main__':

  # Loading the example image for performing video-extensometry
  # This image is distributed with Crappy
  img = crappy.resources.ve_markers

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

  # This VideoExtenso Block calculates the strain of the image by tracking the
  # displacement of spots on the acquired images
  # This Block is actually also the one that generates the fake strain on the
  # image, but that wouldn't be the case in real life
  # It takes the target strain as an input, and outputs both the computed
  # strain and the positions of the tracked spots
  extenso = crappy.blocks.VideoExtenso(
      '',  # The name of Camera to open is ignored because image_generator is
      # given
      config=True,  # Displaying the configuration window before starting,
      # config=False is not implemented yet
      display_images=True,  # The displayer window will allow to follow the
      # spots on the acquired images
      freq=50,  # Lowering the default frequency because it's just a demo
      save_images=False,  # We don't want images to be recorded in this demo
      image_generator=crappy.tool.ApplyStrainToImage(img),  # This argument
      # makes the Block generate fake strain on the given image, only useful
      # for demos
      # The labels for sending the calculated strain to downstream Blocks
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)'),
      white_spots=False,  # We want to detect black spots on a white background
      num_spots=4,  # We want to detect 4 spots on the acquired images

      # Sticking to default for the other arguments
  )

  # This Grapher displays the extension as computed by the VideoExtenso Block
  graph = crappy.blocks.Grapher(('t(s)', 'Exx(%)'))

  # Linking the Blocks together so that each one sends and received the correct
  # information
  # The Generator drives the VideoExtenso, but also takes decision based on its
  # feedback
  crappy.link(gen, extenso)
  crappy.link(extenso, gen)
  crappy.link(extenso, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
