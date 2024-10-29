# coding: utf-8

import numpy as np
from typing import Literal
from ..._global import OptionalModule
from ..camera_config import SpotsBoxes, Box

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DICVETool:
  """This class is the core of the :class:`~crappy.blocks.DICVE` Block.

  It tracks patches on images received from a :class:`~crappy.camera.Camera` 
  object, and computes a strain value at each new image.

  It relies on cross-correlation algorithms to calculate the displacement.
  Different algorithms are available depending on the needs. This tool is
  mainly used to perform video-extensometry on speckled surfaces, although it
  could as well be of use for other applications.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *DISVE* to *DICVETool*
  """

  def __init__(self,
               patches: SpotsBoxes,
               method: Literal['Disflow', 'Lucas Kanade',
                               'Pixel precision', 'Parabola'] = 'Disflow',
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               border: float = 0.2,
               safe: bool = True,
               follow: bool = True) -> None:
    """Sets a few attributes and initializes DISFlow if this method was
    selected.

    Args:
      patches: An instance of the
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` class,
        containing the coordinates of the patches to track.
      method: The method to use to calculate the displacement. `Disflow` uses
        opencv's DISOpticalFlow and `Lucas Kanade` uses opencv's
        calcOpticalFlowPyrLK, while all other methods are based on a basic
        cross-correlation in the Fourier domain. `Pixel precision` calculates
        the displacement by getting the position of the maximum of the
        cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. `Parabola` refines the result of
        `Pixel precision` by interpolating the neighborhood of the maximum, and
        has thus a sub-pixel resolution.

        .. versionadded:: 1.5.9
      alpha: Weight of the smoothness term in DISFlow, as a :obj:`float`.
      delta: Weight of the color constancy term in DISFlow, as a :obj:`float`.
      gamma: Weight of the gradient constancy term in DISFlow , as a
        :obj:`float`.
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DISFlow (`0` means full scale), as an :obj:`int`.
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DISFlow, as an :obj:`int`.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DISFlow, as an :obj:`int`.

        .. versionchanged:: 1.5.9
           renamed from *gditerations* to *gradient_iterations*
      patch_size: Size of an image patch for matching in DISFlow
        (in pixels).
      patch_stride: Stride between neighbor patches in DISFlow. Must be
        less than patch size.
      border: Crop the patch on each side according to this value before
        calculating the displacements. 0 means no cropping, 1 means the entire
        patch is cropped.
      safe: If :obj:`True`, checks whether the patches aren't exiting the
        image, and raises an error if that's the case.
      follow: It :obj:`True`, the patches will move to follow the displacement
        of the image.

    .. versionremoved:: 1.5.10 *img0* and *show_image* arguments
    """

    # These attributes are accessed by the parent class
    self.patches = patches
    self._offsets = [(0, 0) for _ in patches]

    # Other attributes to set
    self._method = method
    self._border = border
    self._safe = safe
    self._follow = follow

    # These attributes will be set later
    self._img0 = None
    self._height, self._width = None, None

    # Initialize DISFlow if it is the selected method
    if self._method == 'Disflow':
      self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
      self._dis.setVariationalRefinementIterations(iterations)
      self._dis.setVariationalRefinementAlpha(alpha)
      self._dis.setVariationalRefinementDelta(delta)
      self._dis.setVariationalRefinementGamma(gamma)
      self._dis.setFinestScale(finest_scale)
      self._dis.setGradientDescentIterations(gradient_iterations)
      self._dis.setPatchSize(patch_size)
      self._dis.setPatchStride(patch_stride)
    else:
      self._dis = None

  def set_img0(self, img0: np.ndarray) -> None:
    """Sets the reference image for the cross-correlation.
    
    .. versionadded:: 1.5.10
    """

    self._img0 = img0
    self._height, self._width, *_ = img0.shape

    # Now that there's an initial image, checking that the patches are valid
    if self._safe:
      self._check_offsets()

  def calculate_displacement(
      self, img: np.ndarray) -> tuple[list[tuple[float, float]], float, float,
                                      list[tuple[float, float]]]:
    """Returns the displacement of every patch, calculated according to the
    chosen method.

    Also updates the patch offsets if required, and updates the window for
    following the patches if any.

    .. versionadded:: 1.5.9
    """

    # Making sure the reference image exists
    if self._img0 is None:
      raise ValueError("The method set_img0 must be called first for setting "
                       "the reference image !")

    # Compute the displacement for each patch
    displacements = []
    for patch, offset in zip(self.patches, self._offsets):

      if patch is None:
        continue

      if self._method == 'Disflow':
        displacements.append(self._calc_disflow(patch, img, offset))

      elif self._method == 'Pixel precision':
        displacements.append(self._calc_pixel_precision(patch, img, offset))

      elif self._method == 'Parabola':
        displacements.append(self._calc_parabola(patch, img, offset))

      elif self._method == 'Lucas Kanade':
        displacements.append(self._calc_lucas_kanade(patch, img, offset))

      else:
        raise ValueError("Wrong method specified !")

    # If required, updates the patch offsets
    if self._follow:
      for disp, (y_offset, x_offset), patch in zip(displacements,
                                                   self._offsets,
                                                   self.patches):

        patch.x_start = round(patch.x_start + disp[0])
        patch.x_end = round(patch.x_end + disp[0])
        patch.y_start = round(patch.y_start + disp[1])
        patch.y_end = round(patch.y_end + disp[1])
        
        patch.x_centroid = (patch.x_end + patch.x_start) / 2
        patch.y_centroid = (patch.y_end + patch.y_start) / 2

        disp[0] += x_offset
        disp[1] += y_offset

        patch.x_disp = disp[0]
        patch.y_disp = disp[1]

      self._offsets = [(round(y_disp), round(x_disp)) for
                       x_disp, y_disp in displacements]

      # Check that the patches are not exiting the image
      if self._safe:
        self._check_offsets()

    else:
      for (x_disp, y_disp), patch in zip(displacements, self.patches):
        patch.x_disp = x_disp
        patch.y_disp = y_disp
    
    max_x = max(self.patches, 
                key=lambda patch_: patch_.x_centroid if patch_ is not None
                else -float('inf'))
    min_x = min(self.patches,
                key=lambda patch_: patch_.x_centroid if patch_ is not None
                else float('inf'))
    max_y = max(self.patches,
                key=lambda patch_: patch_.y_centroid if patch_ is not None
                else -float('inf'))
    min_y = min(self.patches,
                key=lambda patch_: patch_.y_centroid if patch_ is not None
                else float('inf'))

    # If there are multiple spots, the x and y strains can be computed
    if len(self.patches) > 1:
      try:
        exx = ((max_x.x_disp - min_x.x_disp) / self.patches.x_l0) * 100
      except ZeroDivisionError:
        exx = 0
      try:
        eyy = ((max_y.y_disp - min_y.y_disp) / self.patches.y_l0) * 100
      except ZeroDivisionError:
        eyy = 0
      centers = [(patch.y_centroid, patch.x_centroid)
                 for patch in self.patches if patch is not None]
      disps = [(patch.y_disp, patch.x_disp)
               for patch in self.patches if patch is not None]
      return centers, eyy, exx, disps

    # If only one spot was detected, the strain isn't computed
    else:
      x = self.patches[0].x_centroid
      y = self.patches[0].y_centroid
      x_disp = self.patches[0].x_disp
      y_disp = self.patches[0].y_disp
      return [(y, x)], 0, 0, [(y_disp, x_disp)]

  def _calc_disflow(self,
                    patch: Box,
                    img: np.ndarray,
                    offset: tuple[int, int]) -> list[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using DISFlow."""

    disp_img = self._dis.calc(self._get_patch(self._img0, patch, offset),
                              self._get_patch(img, patch), None)
    return np.average(self._trim_patch(disp_img), axis=(0, 1)).tolist()

  def _calc_pixel_precision(self,
                            patch: Box,
                            img: np.ndarray,
                            offset: tuple[int, int]) -> list[float]:
    """Returns the displacement between the original and the current image with
    a precision limited to 1 pixel."""

    cross_correl, max_width, max_height = self._cross_correlation(
      self._get_patch(self._img0, patch, offset), self._get_patch(img, patch))

    height, width = cross_correl.shape[0], cross_correl.shape[1]
    return [-(max_width - width / 2), -(max_height - height / 2)]

  def _calc_parabola(self,
                     patch: Box,
                     img: np.ndarray,
                     offset: tuple[int, int]) -> list[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using two parabola fits (one in x and one in y)."""

    cross_correl, max_width, max_height = self._cross_correlation(
      self._get_patch(self._img0, patch, offset), self._get_patch(img, patch))

    height, width = cross_correl.shape[0], cross_correl.shape[1]
    y_disp = -(max_height - height / 2)
    x_disp = -(max_width - width / 2)

    x_disp -= self._parabola_fit(cross_correl[max_height,
                                 max_width - 1: max_width + 2])
    y_disp -= self._parabola_fit(cross_correl[max_height - 1: max_height + 2,
                                 max_width])

    return [x_disp, y_disp]

  def _calc_lucas_kanade(self,
                         patch: Box,
                         img: np.ndarray,
                         offset: tuple[int, int]) -> list[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using the Lucas Kanade algorithm."""

    # Getting the center of the patch
    x_top, x_bottom, y_left, y_right = patch.sorted()
    center_y, center_x = (y_right - y_left) // 2, (x_bottom - x_top) // 2

    next_, _, _ = cv2.calcOpticalFlowPyrLK(
      self._get_patch(self._img0, patch, offset), self._get_patch(img, patch),
      np.array([[center_x, center_y]]).astype('float32'), None)
    new_x, new_y = np.squeeze(next_)

    return [new_x - center_x, new_y - center_y]

  @staticmethod
  def _parabola_fit(arr: np.ndarray) -> float:
    """Returns the abscissa of the maximum of a parabola defined by 3
    points in positions x=-1, x=0 and x=1.

    Args:
      arr: This array contains the y values for the 3 points.
    """

    return float((arr[0] - arr[2]) / (2 * (arr[0] - 2 * arr[1] + arr[2])))

  @staticmethod
  def _cross_correlation(img0: np.ndarray,
                         img1: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Performs a cross-correlation operation on two patches in the Fourier
    domain.

    Returns:
      The result of the correlation in the real domain as an image, as well as
      the position of the maximum of this image.
    """

    # Getting the size of the image
    height, width = img0.shape[0], img0.shape[1]

    # Find the closest power of 2 length from the image
    height_log2 = 2 ** int(np.log2(height + 0.5))
    width_log2 = 2 ** int(np.log2(width + 0.5))

    # Cropping the image to a power of 2
    img0 = img0[(height - height_log2) // 2: (height + height_log2) // 2,
                (width - width_log2) // 2: (width + width_log2) // 2]
    img1 = img1[(height - height_log2) // 2: (height + height_log2) // 2,
                (width - width_log2) // 2: (width + width_log2) // 2]

    # Convert to Fourier for fast cross-correlation
    img0_fourier = cv2.dft(img0.astype(np.float32),
                           flags=cv2.DFT_COMPLEX_OUTPUT)
    img1_fourier = cv2.dft(img1.astype(np.float32),
                           flags=cv2.DFT_COMPLEX_OUTPUT)

    # Compute cross-correlation by convolution
    cross_fourier = cv2.mulSpectrums(img0_fourier, img1_fourier,
                                     flags=0, conjB=True)

    # Convert back to physical space
    cross_shifted = cv2.idft(cross_fourier)
    # Un-shift after FFT
    cross_correl = np.fft.ifftshift(cross_shifted[:, :, 0])

    # Find the maximum of the cross-correlation
    max_width, max_height = cv2.minMaxLoc(cross_correl)[-1]
    # Keep the average in case several maxima found
    max_width, max_height = int(np.mean(max_width)), int(np.mean(max_height))

    return cross_correl, max_width, max_height

  @staticmethod
  def _get_patch(img: np.ndarray,
                 patch: Box,
                 offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Returns the part of the image corresponding to the given patch at the
    given offset."""

    y_off, x_off = offset
    x_top, x_bottom, y_left, y_right = patch.sorted()
    return np.array(img[y_left - y_off: y_right - y_off,
                        x_top - x_off: x_bottom - x_off])

  def _trim_patch(self, patch: np.ndarray) -> np.ndarray:
    """Trims the border of a patch according to the value set by the user, and
    returns the sub image corresponding to the trimmed patch."""

    height, width, *_ = patch.shape
    return patch[int(height * self._border / 2):
                 int(height * (1 - self._border / 2)),
                 int(width * self._border / 2):
                 int(width * (1 - self._border / 2))]

  def _check_offsets(self) -> None:
    """Check if the patches are still within the image, and raises an error if
    one of them is out."""

    for patch in self.patches:

      if patch is None:
        continue

      x_top, x_bottom, y_left, y_right = patch.sorted()

      # Checking the left border
      if x_top < 0:
        raise RuntimeError("Region exiting the ROI (left)")

      # Checking the right border
      elif x_bottom > self._width:
        raise RuntimeError("Region exiting the ROI (right)")

      # Checking the top border
      if y_left < 0:
        raise RuntimeError("Region exiting the ROI (top)")

      # Checking the bottom border
      elif y_right > self._height:
        raise RuntimeError("Region exiting the ROI (bottom)")
