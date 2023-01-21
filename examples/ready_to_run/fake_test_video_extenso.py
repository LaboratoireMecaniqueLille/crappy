# coding: utf-8

"""
Demonstration of a Videoextensometry controlled test.

This program is intended as a demonstration and is fully virtual.

No hardware required
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


def elastic_law(_: float) -> float:
  """No elastic law in this simple example."""

  return 0.


if __name__ == "__main__":
  img = crappy.resources.ve_markers

  speed = 5 / 60  # mm/sec

  # Load until the strain is reached, then unload until force is 0
  generator = crappy.blocks.Generator(path=sum([[
    {'type': 'Constant', 'value': speed,
     'condition': 'Exx(%)>{}'.format(5 * i)},
    {'type': 'Constant', 'value': -speed, 'condition': 'F(N)<0'}]
    for i in range(1, 5)], []), spam=False)

  # Our fake machine
  machine = crappy.blocks.FakeMachine(rigidity=5000, l0=20, max_strain=17,
                                      sigma={'F(N)': 0.5},
                                      plastic_law=elastic_law)

  crappy.link(generator, machine)
  crappy.link(machine, generator)

  # The block performing the video-extensometry
  ve = crappy.blocks.VideoExtenso('',
                                  display_images=True,
                                  blur=False,
                                  image_generator=Apply_strain_img(img),
                                  verbose=True)
  # This modifier will generate an image with the values of strain
  # coming from the FakeMachine block
  crappy.link(machine, ve)

  graph_def2 = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(ve, graph_def2, modifier=crappy.modifier.Mean(10))

  crappy.start()
