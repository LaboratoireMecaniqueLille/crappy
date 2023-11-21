# coding: utf-8

import numpy as np

from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule('opencv-python')


class ApplyStrainToImage:
  """This class reshapes an image depending on input strain values. It is meant
  to simulate the stretching of a sample during a tensile test.

  Its main use case is for generating example scripts that do not require any
  hardware to run.
  
  .. versionadded:: 2.0.0
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
