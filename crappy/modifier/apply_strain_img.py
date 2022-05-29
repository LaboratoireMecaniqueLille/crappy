# coding: utf-8

import numpy as np
from typing import Dict, Any
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

from .modifier import Modifier


class Apply_strain_img(Modifier):
  """This modifier reads the strain values along X and Y (in %) and returns an
  image deformed according to these values."""

  def __init__(self,
               img: np.ndarray,
               exx_label: str = 'Exx(%)',
               eyy_label: str = 'Eyy(%)',
               img_label: str = 'frame') -> None:
    """Sets the args and initializes the parent class.

    Args:
      img: The base image to be deformed, as a :mod:`numpy` array.
      exx_label: The labels carrying the strain value to apply in the X
        direction.
      eyy_label: The labels carrying the strain value to apply in the Y
        direction.
      img_label: The label carrying the deformed image.
    """

    super().__init__()

    # Setting the args
    self._img = img
    self._exx_label = exx_label
    self._eyy_label = eyy_label
    self._img_label = img_label

    # Building the lookup arrays for the cv2.remap method
    height, width, *_ = img.shape
    orig_x, orig_y = np.meshgrid(range(width), range(height))
    # These arrays correspond to the original state of the image
    self._orig_x = orig_x.astype(np.float32)
    self._orig_y = orig_y.astype(np.float32)

    # These arrays are meant to be added to the original image ones
    # If added as is, they correspond to a 100% strain state in both directions
    self._x_strain = self._orig_x * width / (width - 1) - width / 2
    self._y_strain = self._orig_y * height / (height - 1) - height / 2

  def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Reads the X and Y strain values sent through the link, resizes the image
    accordingly and returns it along with the received data."""

    exx, eyy = data[self._exx_label] / 100, data[self._eyy_label] / 100

    # The final lookup table is the sum of the original state ones plus the
    # 100% strain one weighted by a ratio
    transform_x = self._orig_x - (exx / (1 + exx)) * self._x_strain
    transform_y = self._orig_y - (eyy / (1 + eyy)) * self._y_strain

    data[self._img_label] = cv2.remap(self._img, transform_x, transform_y, 1)
    return data
