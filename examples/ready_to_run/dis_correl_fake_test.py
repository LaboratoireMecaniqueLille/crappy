# coding: utf-8

"""
Demonstration of a DIC controlled test.

This program is intended as a demonstration and is fully virtual.

No hardware required.
Requires the cv2 module to be installed.
"""

import crappy
import numpy as np
import cv2


class Apply_strain_img:
  """This class reshapes an image depending on input strain values."""

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


def plastic_law(_: float) -> float:
  """No elastic law in this simple example."""

  return 0.


if __name__ == "__main__":

  # The image used for replacing the camera
  img = crappy.resources.speckle

  speed = 5 / 60  # mm/sec

  # Load until the strain is reached, then unload until force is 0
  generator = crappy.blocks.Generator(path=sum([[
    {'type': 'Constant', 'value': speed, 'condition': f'Exx(%)>{5 * i}'},
    {'type': 'Constant', 'value': -speed, 'condition': 'F(N)<0'}]
    for i in range(1, 5)], []), spam=False)

  # The Block emulating the behavior of a tensile test machine
  machine = crappy.blocks.FakeMachine(max_strain=17, k=5000, l0=20,
                                      plastic_law=plastic_law,
                                      sigma={'F(N)': 0.5})

  crappy.link(generator, machine)
  crappy.link(machine, generator)

  # The Block performing the DIC
  dis = crappy.blocks.DISCorrel('', display_images=True,
                                labels=['t(s)', 'meta', 'x', 'y',
                                        'measured_Exx(%)', 'measured_Eyy(%)'],
                                image_generator=Apply_strain_img(img),
                                verbose=True)
  crappy.link(machine, dis)

  # The Block displaying the measured strain
  graph_def2 = crappy.blocks.Grapher(('t(s)', 'measured_Exx(%)'),
                                     ('t(s)', 'measured_Eyy(%)'))
  crappy.link(dis, graph_def2)

  # Starting the test
  crappy.start()
