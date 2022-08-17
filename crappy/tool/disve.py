# coding: utf-8

import numpy as np
from typing import List, Tuple
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DISVE:
  """This tool computes the displacement of regions of interest (patches) on an
  image compared to a reference image.

  It relies on cross-correlation algorithms to calculate the displacement.
  Different algorithms are available depending on the needs.
  This tool is mainly used to perform video-extensometry on speckled surfaces,
  although it can as well be of use for other applications.
  """

  def __init__(self,
               img0: np.ndarray,
               patches: List[Tuple[int, int, int, int]],
               method: str = 'Disflow',
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
               follow: bool = True,
               show_image: bool = False) -> None:
    """Sets a few attributes and initializes Disflow if this method was
    selected.

    Args:
      img0: The reference image for the cross-correlation.
      patches: The regions to track, should be a tuple of 4 values
        (pos y, pos x, height, width).
      method: The method to use to calculate the displacement. `Disflow` uses
        opencv's DISOpticalFlow and `Lucas Kanade` uses opencv's
        calcOpticalFlowPyrLK, while all other methods are based on a basic
        cross-correlation in the Fourier domain. `Pixel precision` calculates
        the displacement by getting the position of the maximum of the
        cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. `Parabola` refines the result of
        `Pixel precision` by interpolating the neighborhood of the maximum, and
        have thus sub-pixel resolutions.
      alpha: Setting for Disflow.
      delta: Setting for Disflow.
      gamma: Setting for Disflow.
      finest_scale: The last scale for Disflow (`0` means full scale).
      iterations: Variational refinement iterations for Disflow.
      gradient_iterations: Gradient descent iterations for Disflow.
      patch_size: Correlation patch size for Disflow.
      patch_stride: Correlation patch stride for Disflow.
      border: Crop the patch on each side according to this value before
        calculating the displacements. 0 means no cropping, 1 means the entire
        patch is cropped.
      safe: Checks whether the patches aren't exiting the image, and raises an
        error if that's the case.
      follow: It :obj:`True`, the patches will move to follow the displacement
        of the image.
      show_image: If :obj:`True`, displays the real-time position of the
        patches on the image. This feature is mainly meant for debugging.
    """

    self._img0 = img0
    self._patches = patches
    self._method = method
    self._height, self._width = img0.shape
    self._border = border
    self._safe = safe
    self._follow = follow
    self._show_image = show_image

    # Initialize Disflow if it is the selected method
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

    self._offsets = [(0, 0) for _ in self._patches]
    if self._safe:
      self._check_offsets()

    # Initializing the window for following the patches
    if self._show_image:
      cv2.namedWindow('DISVE', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

  def calculate_displacement(self, img: np.ndarray) -> List[float]:
    """Returns the displacement of every patch, calculated according to the
    chosen method.

    Also updates the patch offsets if required, and updates the window for
    following the patches if any.
    """

    # first, compute the displacement for each patch
    displacements = []
    for patch, offset in zip(self._patches, self._offsets):

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
      for disp, (y_offset, x_offset) in zip(displacements, self._offsets):
        disp[0] += x_offset
        disp[1] += y_offset

      self._offsets = [(int(y_disp), int(x_disp)) for
                       x_disp, y_disp in displacements]

      # Check that the patches are not exiting the image
      if self._safe:
        self._check_offsets()

    # Update the display
    if self._show_image:
      self._update_img(img)

    displacements = [coord for disp in displacements for coord in disp]
    return displacements

  def close(self) -> None:
    """Closes the window for following the patches."""

    if self._show_image:
      cv2.destroyWindow("DISVE")

  def _calc_disflow(self,
                    patch: Tuple[int, int, int, int],
                    img: np.ndarray,
                    offset: Tuple[int, int]) -> List[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using Disflow."""

    disp_img = self._dis.calc(self._get_patch(self._img0, patch),
                              self._get_patch(img, patch, offset),
                              None)
    return np.average(self._trim_patch(disp_img), axis=(0, 1)).tolist()

  def _calc_pixel_precision(self,
                            patch: Tuple[int, int, int, int],
                            img: np.ndarray,
                            offset: Tuple[int, int]) -> List[float]:
    """Returns the displacement between the original and the current image with
    a precision limited to 1 pixel."""

    cross_correl, max_width, max_height = self._cross_correlation(
      self._get_patch(self._img0, patch), self._get_patch(img, patch, offset))

    height, width = cross_correl.shape[0], cross_correl.shape[1]
    return [-(max_width - width / 2), -(max_height - height / 2)]

  def _calc_parabola(self,
                     patch: Tuple[int, int, int, int],
                     img: np.ndarray,
                     offset: Tuple[int, int]) -> List[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using two parabola fits (one in x and one in y)."""

    cross_correl, max_width, max_height = self._cross_correlation(
      self._get_patch(self._img0, patch), self._get_patch(img, patch, offset))

    height, width = cross_correl.shape[0], cross_correl.shape[1]
    y_disp = -(max_height - height / 2)
    x_disp = -(max_width - width / 2)

    x_disp -= self._parabola_fit(cross_correl[max_height,
                                 max_width - 1: max_width + 2])
    y_disp -= self._parabola_fit(cross_correl[max_height - 1: max_height + 2,
                                 max_width])

    return [x_disp, y_disp]

  def _calc_lucas_kanade(self,
                         patch: Tuple[int, int, int, int],
                         img: np.ndarray,
                         offset: Tuple[int, int]) -> List[float]:
    """Returns the displacement between the original and the current image with
    a sub-pixel precision, using the Lucas Kanade algorithm."""

    # Getting the center of the patch
    center_y, center_x = patch[2] // 2, patch[3] // 2

    next_, _, _ = cv2.calcOpticalFlowPyrLK(
      self._get_patch(self._img0, patch), self._get_patch(img, patch, offset),
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

    return (arr[0] - arr[2]) / (2 * (arr[0] - 2 * arr[1] + arr[2]))

  @staticmethod
  def _cross_correlation(img0: np.ndarray,
                         img1: np.ndarray) -> Tuple[np.ndarray, int, int]:
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
                 patch: Tuple[int, int, int, int],
                 offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Returns the part of the image corresponding to the given patch at the
    given offset."""

    (y_min, x_min, height, width), (y_offset, x_offset) = patch, offset
    return np.array(img[y_min + y_offset:y_min + height + y_offset,
                        x_min + x_offset:x_min + width + x_offset])

  def _trim_patch(self, patch: np.ndarray) -> np.ndarray:
    """Trims the border of a patch according to the value set by the user, and
    returns the tuple corresponding to the trimmed patch."""

    height, width, *_ = patch.shape
    return patch[int(height * self._border / 2):
                 int(height * (1 - self._border / 2)),
                 int(width * self._border / 2):
                 int(width * (1 - self._border / 2))]

  def _check_offsets(self) -> None:
    """Check if the patches are still within the image, and raises an error if
    one of them is out."""

    for (y_min, x_min, height, width), \
            (y_offset, x_offset) in zip(self._patches, self._offsets):

      # Checking the left border
      if x_offset + x_min < 0:
        raise RuntimeError("Region exiting the ROI (left)")

      # Checking the right border
      elif x_offset + x_min + width > self._width:
        raise RuntimeError("Region exiting the ROI (right)")

      # Checking the top border
      if y_offset + y_min < 0:
        raise RuntimeError("Region exiting the ROI (top)")

      # Checking the bottom border
      elif y_offset + y_min + height > self._height:
        raise RuntimeError("Region exiting the ROI (bottom)")

  def _update_img(self, img: np.ndarray) -> None:
    """Updates the display of the window for following thr patches."""

    for (y_min, x_min, height, width), \
            (y_offset, x_offset) in zip(self._patches, self._offsets):

      img[y_min + y_offset:y_min + y_offset + 1,
          x_min + x_offset: x_min + x_offset + width] = 255
      img[y_min + height + y_offset:y_min + height + y_offset + 1,
          x_min + x_offset: x_min + x_offset + width] = 255
      img[y_min + y_offset: y_min + height + y_offset,
          x_min + x_offset:x_min + x_offset + 1] = 255
      img[y_min + y_offset: y_min + height + y_offset,
          x_min + height + x_offset:x_min + height + x_offset + 1] = 255

    cv2.imshow("DISVE", img)
    cv2.waitKey(5)
