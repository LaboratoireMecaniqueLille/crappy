# coding: utf-8

"""
This example demonstrates the use of the DICVE Block in the simplest possible
use case. It does not require any hardware to run.

This Block computes the strain on acquired images by tracking several patches
using digital image correlation techniques. It outputs the computed strain as
well as the position and displacement of the patches.

In this example, a fake strain is generated on a static image of a sample with
a speckle. The level of strain is controlled by a Generator Block, and applied
to the images by the DICVE Block. This same DICVE Block then calculates the
strain on the images, and outputs it to a Grapher Block for display.

After starting this script, you have to select the patches to track in the
configuration window. Adjust the size of the patches, apply the settings, then
draw the patches to track by left-clicking and dragging. Then, close the
configuration window and watch the strain be calculated in real time. This demo
normally ends automatically after 2 minutes, but can also be stopped by hitting
CTRL+C.
"""

import crappy
import numpy as np
import cv2


class ApplyStrainToImage:
  """This class reshapes an image depending on input strain values. It is meant
  to simulate the stretching of a sample during a tensile test.

  You don't have to care too much about what it contains, it is not needed for
  understanding the example.
  """

  def __init__(self,
               image: np.ndarray) -> None:
    """Sets the base image and initializes the necessary objects to use."""

    self._img = image

    # Building the lookup arrays for the cv2.remap method
    height, width, *_ = image.shape
    orig_x, orig_y = np.meshgrid(range(width), range(height))
    # These arrays correspond to the original state of the image
    self._orig_x = orig_x.astype(np.float32)
    self._orig_y = orig_y.astype(np.float32)

    # These arrays are meant to be added to the original image ones
    # If added as is, they correspond to a 100% strain state in both directions
    self._x_strain = self._orig_x * width / (width - 1) - width / 2
    self._y_strain = self._orig_y * height / (height - 1) - height / 2

  def __call__(self, exx: float, eyy: float) -> np.ndarray:
    """Returns the reshaped image, based on the given strain values."""

    exx /= 100
    eyy /= 100

    # The final lookup table is the sum of the original state ones plus the
    # 100% strain one weighted by a ratio
    transform_x = self._orig_x - (exx / (1 + exx)) * self._x_strain
    transform_y = self._orig_y - (eyy / (1 + eyy)) * self._y_strain

    return cv2.remap(self._img, transform_x, transform_y, 1)


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
      config=True,  # Displaying the configuration window before starting,
      # mandatory if the patches to track ar not given as arguments
      display_images=True,  # The displayer window will allow to follow the
      # patches on the speckle image
      freq=50,  # Lowering the default frequency because it's just a demo
      save_images=False,  # We don't want images to be recorded in this demo
      image_generator=ApplyStrainToImage(img),  # This argument makes the Block
      # generate fake strain on the given image, only useful for demos
      patches=None,  # The patches to track are not provided here, so they must
      # be selected in the configuration window
      # The labels for sending the calculated strain to downstream Blocks
      labels=('t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'Disp(px)'),
      method='Disflow',  # The default image correlation method

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
