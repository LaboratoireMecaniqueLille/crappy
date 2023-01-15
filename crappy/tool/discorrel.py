# coding: utf-8

from typing import List, Optional

from .._global import OptionalModule
from .camera_config import Box
from .fields import get_res, get_field, allowed_fields

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
import numpy as np


class DISCorrel:
  """This class simply stores the data needed for performing image correlation
  using DisFlow, and also performs the correlation.

  Multiple parameters of DisFlow can be tuned, but not all of them. The class
  also takes care of projecting the image on the chosen base of fields, and
  optionally to estimate the residuals of the correlation process.
  """

  def __init__(self,
               fields: Optional[List[str]] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               init: bool = True,
               iterations: int = 1,
               gradient_iterations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               box: Optional[Box] = None) -> None:
    """Sets the parameters of DisFlow.

    Args:
      fields: The base of fields to use for the projection, given as a
        :obj:`list` of :obj:`str`. The available fields are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

      alpha: Weight of the smoothness term in DisFlow.
      delta: Weight of the color constancy term in DisFlow.
      gamma: Weight of the gradient constancy term in DisFlow
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DisFlow (`0` means full scale).
      init: If :obj:`True`, the last field is used to initialize the
        calculation for the next one.
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DisFlow.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DisFlow.
      patch_size: Size of an image patch for matching in DisFlow
        (in pixels).
      patch_stride: Stride between neighbor patches in DisFlow. Must be
        less than patch size.
    """

    if fields is not None and not all((field in allowed_fields
                                       for field in fields)):
      raise ValueError(f"The only allowed values for the fields "
                       f"are {allowed_fields}")
    self._fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    self._init = init

    # These attributes will be set later
    self._img0 = None
    self._height, self._width = None, None
    self.box = None
    self._dis_flow = None
    self._base = None
    self._norm2 = None

    if box is not None:
      self.set_box(box)

    # Setting the parameters of Disflow
    self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self._dis.setVariationalRefinementAlpha(alpha)
    self._dis.setVariationalRefinementDelta(delta)
    self._dis.setVariationalRefinementGamma(gamma)
    self._dis.setFinestScale(finest_scale)
    self._dis.setVariationalRefinementIterations(iterations)
    self._dis.setGradientDescentIterations(gradient_iterations)
    self._dis.setPatchSize(patch_size)
    self._dis.setPatchStride(patch_stride)

  def set_img0(self, img0: np.ndarray) -> None:
    """Sets the initial image to use for the correlation."""

    self._img0 = img0
    self._height, self._width, *_ = img0.shape
    self._dis_flow = np.zeros((self._height, self._width, 2))

  def set_box(self, box: Box) -> None:
    """Sets the region of interest to use for the correlation, and initializes
    other attributes."""

    # Sets the bounding box
    self.box = box
    x_top, x_bottom, y_left, y_right = box.sorted()
    box_height = y_right - y_left
    box_width = x_bottom - x_top

    # Creates and populates the base fields to use for correlation
    fields = np.empty((box_height, box_width, 2, len(self._fields)),
                      dtype=np.float32)
    for i, string in enumerate(self._fields):
      fields[:, :, 0, i], fields[:, :, 1, i] = get_field(string, box_height,
                                                         box_width)

    # These attributes will be used later
    self._base = [fields[:, :, :, i] for i in range(fields.shape[3])]
    self._norm2 = [np.sum(base_field ** 2) for base_field in self._base]

  def get_data(self,
               img: np.ndarray,
               residuals: bool = False) -> List[float]:
    """Processes the input image and returns the required data in a
    :obj:`list`.

    Args:
      img: The new image to process.
      residuals: Whether the residuals should be calculated or not for the
        image.

    Returns:
      A :obj:`list` containing the data to calculate, and the residuals at the
      end if requested.
    """

    # Making sure the reference image and the base fields were set
    if self._img0 is None:
      raise ValueError("The method set_img0 must be called first for setting "
                       "the reference image !")
    elif self._base is None:
      raise ValueError("The method set_box must be called first for setting "
                       "the region of interest !")

    # Updating the optical flow with the latest image
    if self._init:
      self._dis_flow = self._dis.calc(self._img0, img, self._dis_flow)
    else:
      self._dis_flow = self._dis.calc(self._img0, img, None)

    # Getting the values to calculate as floats
    ret = [np.sum(vec * self._crop(self._dis_flow)) / n2 for vec, n2 in
           zip(self._base, self._norm2)]

    # Adding the average residual value if requested
    if residuals:
      ret.append(np.average(np.abs(get_res(self._img0, img, self._dis_flow))))

    return ret

  def _crop(self, img: np.ndarray) -> np.ndarray:
    """Crops the image to the given region of interest."""

    x_min, x_max, y_min, y_max = self.box.sorted()
    return img[y_min:y_max, x_min:x_max]
