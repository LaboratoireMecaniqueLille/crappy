# coding: utf-8

from typing import Literal
import numpy as np

from ..._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

allowed_fields = ('x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z')


def get_field(field_string: Literal['x', 'y', 'r', 'exx', 'eyy',
                                    'exy', 'eyx', 'exy2', 'z'],
              h: int,
              w: int) -> tuple[np.ndarray, np.ndarray]:
  """Creates and returns the two fields on which the image will be projected,
  as :mod:`numpy` arrays.
  
  This function is used by the
  :class:`~crappy.tool.image_processing.DISCorrelTool` and
  :class:`~crappy.tool.image_processing.GPUCorrelTool` tools.

  Args:
    field_string: The :obj:`str` describing the field on which to project the
      image for correlation.

      .. versionchanged:: 1.5.10 renamed from *s* to *field_string*
    h: The height of the image, as an :obj:`int`.
    w: The width of the image, as an :obj:`int`.
  
  .. versionadded:: 1.4.0
  """

  if field_string == 'x':
    return (np.ones((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32))

  elif field_string == 'y':
    return (np.zeros((h, w), dtype=np.float32),
            np.ones((h, w), dtype=np.float32))

  elif field_string == 'r':
    u, v = np.meshgrid(np.linspace(-w, w, w, dtype=np.float32),
                       np.linspace(-h, h, h, dtype=np.float32))
    return v * np.pi / 360, -u * np.pi / 360

  elif field_string == 'exx':
    return (np.tile(np.linspace(-w / 200, w / 200, w, dtype=np.float32),
                    (h, 1)),
            np.zeros((h, w), dtype=np.float32))

  elif field_string == 'eyy':
    return (np.zeros((h, w), dtype=np.float32),
            np.swapaxes(np.tile(np.linspace(-h / 200, h / 200, h,
                                            dtype=np.float32), (w, 1)), 1, 0))

  elif field_string == 'exy':
    return (np.swapaxes(np.tile(np.linspace(-h / 200, h / 200, h,
                                            dtype=np.float32), (w, 1)), 1, 0),
            np.zeros((h, w), dtype=np.float32))

  elif field_string == 'eyx':
    return (np.zeros((h, w), dtype=np.float32),
            np.tile(np.linspace(-w / 200, w / 200, w, dtype=np.float32),
                    (h, 1)))

  elif field_string == 'exy2':
    return (np.swapaxes(np.tile(np.linspace(-h / 200, h / 200, h,
                                            dtype=np.float32), (w, 1)), 1, 0),
            np.tile(np.linspace(-w / 200, w / 200, w, dtype=np.float32),
                    (h, 1)))

  elif field_string == 'z':
    u, v = np.meshgrid(np.linspace(-w, w, w, dtype=np.float32),
                       np.linspace(-h, h, h, dtype=np.float32))
    return u / 200, v / 200

  else:
    raise NameError(f"Unknown field {field_string}")


def get_res(ref: np.ndarray, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
  """Calculates the difference between the original image and the one
  reconstructed from the current image and the calculated flow.

  This function is used by the
  :class:`~crappy.tool.image_processing.DISCorrelTool` tool.

  Args:
    ref: The reference image for calculating the optical flow.

      .. versionchanged:: 1.5.10 renamed from *r* to *ref*
    img: The current image for calculating the optical flow.

      .. versionchanged:: 1.5.10 renamed from *a* to *img*
    flow: The calculated optical flow

      .. versionchanged:: 1.5.10 renamed from *b* to *flow*
    
  .. versionadded:: 1.4.0
  """

  x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
  return ref - cv2.remap(img.astype(np.float32),
                         (x + flow[:, :, 0]).astype(np.float32),
                         (y + flow[:, :, 1]).astype(np.float32), 1)
