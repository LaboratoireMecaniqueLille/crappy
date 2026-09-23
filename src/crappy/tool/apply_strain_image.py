# coding: utf-8

import numpy as np

from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule('opencv-python')


class ApplyStrainToImage:
  """Deforms an image according to horizontal and vertical strain values.

  Instances are callable and can be passed as the ``image_generator`` of a
  :class:`~crappy.blocks.vision.CameraSource` or
  :class:`~crappy.blocks.Camera`. This supports image-processing examples and
  tests without a physical camera.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self,
               image: np.ndarray) -> None:
    """Prepares the lookup arrays used to deform the reference image.

    Args:
      image: Two- or three-dimensional reference image.
    """

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
    """Returns the reference image deformed by the requested strains.

    Args:
      exx: Horizontal strain in percent.
      eyy: Vertical strain in percent.

    Returns:
      Deformed image with the same shape and data type as the reference.
    """

    exx /= 100
    eyy /= 100

    # The final lookup table is the sum of the original state ones plus the
    # 100% strain one weighted by a ratio
    transform_x = self._orig_x - (exx / (1 + exx)) * self._x_strain
    transform_y = self._orig_y - (eyy / (1 + eyy)) * self._y_strain

    return cv2.remap(self._img, transform_x, transform_y, 1)
