# coding: utf-8

from typing import Optional
import logging
from multiprocessing import current_process
import numpy as np
from itertools import combinations

from .box import Box
from .spots_boxes import SpotsBoxes
from ...._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
try:
  from skimage.filters import threshold_otsu
  from skimage.morphology import label
  from skimage.measure import regionprops
except (ModuleNotFoundError, ImportError):
  label = OptionalModule("skimage", "Please install scikit-image to use"
                         "Video-extenso")
  threshold_otsu = regionprops = label


class SpotsDetector:
  """"""

  def __init__(self,
               white_spots: bool = False,
               num_spots: Optional[int] = None,
               min_area: int = 150,
               blur: Optional[int] = 5,
               update_thresh: bool = False,
               safe_mode: bool = False,
               border: int = 5) -> None:
    """

    Args:
      white_spots:
      num_spots:
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of camera and
        the size of the spots to detect.
      blur:
    """

    self.white_spots = white_spots
    self.thresh = 0 if white_spots else 255
    self.min_area = min_area
    self.blur = blur
    self.update_thresh = update_thresh
    self.safe_mode = safe_mode
    self.border = border

    self._logger = logging.getLogger(f"{current_process().name}."
                                     f"{type(self).__name__}")

    if num_spots is not None and num_spots not in range(1, 5):
      raise ValueError("num_spots should be either None, 1, 2, 3 or 4 !")
    self.num_spots = num_spots

    self.spots = SpotsBoxes()

  def detect_spots(self,
                   img: np.ndarray,
                   y_orig: int,
                   x_orig: int) -> None:
    """Transforms the image to improve spot detection, detects up to 4 spots
    and return a Spot_boxes object containing all the detected spots.

    Args:
      img: The sub-image on which the spots should be detected.
      y_orig: The y coordinate of the top-left pixel of the sub-image on the
        entire image.
      x_orig: The x coordinate of the top-left pixel of the sub-image on the
        entire image.

    Returns:
      A Spot_boxes object containing all the detected spots.
    """

    # First, blurring the image if asked to
    if self.blur is not None and self.blur > 1:
      img = cv2.medianBlur(img, self.blur)

    # Determining the best threshold for the image
    self.thresh = threshold_otsu(img)

    # Getting all pixels superior or inferior to threshold
    if self.white_spots:
      black_white = img > self.thresh
    else:
      black_white = img <= self.thresh

    # The original image is not needed anymore
    del img

    # Detecting the spots on the image
    props = regionprops(label(black_white))

    # The black and white image is not needed anymore
    del black_white

    # Removing regions that are too small or not circular
    props = [prop for prop in props if prop.solidity > 0.8
             and prop.area > self.min_area]

    # Detecting overlapping spots and storing the ones that should be removed
    to_remove = list()
    for prop_1, prop_2 in combinations(props, 2):
      if self._overlap_bbox(prop_1, prop_2):
        to_remove.append(prop_2 if prop_1.area > prop_2.area else prop_1)

    # Now removing the overlapping spots
    if to_remove:
      self._logger.log(logging.WARNING, "Overlapping spots found, removing the"
                                        " smaller overlapping ones")
    for prop in to_remove:
      props.remove(prop)

    # Sorting the regions by area
    props = sorted(props, key=lambda prop: prop.area, reverse=True)

    # Keeping only the biggest areas to match the target number of spots
    if self.num_spots is not None:
      props = props[:self.num_spots]
    else:
      props = props[:4]

    # Indicating the user if not enough spots were found
    if not props:
      self._logger.log(logging.WARNING, "No spots found !")
      return
    elif self.num_spots is not None and len(props) != self.num_spots:
      self._logger.log(logging.WARNING, f"Expected {self.num_spots} spots, "
                                        f"found only {len(props)}")
      return

    # Replacing the previously detected spots with the new ones
    self.spots.reset()
    for i, prop in enumerate(props):
      # Extracting the properties of interest
      y, x = prop.centroid
      y_min, x_min, y_max, x_max = prop.bbox
      # Adjusting with the offset given as argument
      x += x_orig
      x_min += x_orig
      x_max += x_orig
      y += y_orig
      y_min += y_orig
      y_max += y_orig

      self.spots[i] = Box(x_start=x_min, x_end=x_max,
                          y_start=y_min, y_end=y_max,
                          x_centroid=x, y_centroid=y)

  @staticmethod
  def _overlap_bbox(prop_1, prop_2) -> bool:
    """Determines whether two bboxes are overlapping or not."""

    y_min_1, x_min_1, y_max_1, x_max_1 = prop_1.bbox
    y_min_2, x_min_2, y_max_2, x_max_2 = prop_2.bbox

    return max((min(x_max_1, x_max_2) - max(x_min_1, x_min_2)), 0) * max(
      (min(y_max_1, y_max_2) - max(y_min_1, y_min_2)), 0) > 0
