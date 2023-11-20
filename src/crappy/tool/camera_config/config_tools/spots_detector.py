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
  """This class detects round spots on a grey level image.

  It takes an image from a :class:`~crappy.tool.camera_config.CameraConfig`
  window as an input of the :meth:`detect_spots` method, and tries to detect
  the requested number of spots on it. It then stores the position and size of
  the detected spots, to pass them later on to the
  :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` along
  with other variables once the CameraConfig window is closed.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               white_spots: bool = False,
               num_spots: Optional[int] = None,
               min_area: int = 150,
               blur: Optional[int] = 5,
               update_thresh: bool = False,
               safe_mode: bool = False,
               border: int = 5) -> None:
    """Sets the arguments.

    Args:
      white_spots: If :obj:`True`, detects white spots over a black background.
        If :obj:`False`, detects black spots over a white background. Also
        passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.
      num_spots: The number of spots to detect, as an :obj:`int` between `1`
        and `4`. If given, will try to detect exactly that number of spots and
        will fail if not enough spots can be detected. If left to :obj:`None`,
        will detect up to `4` spots, but potentially fewer.
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of the camera
        and the size of the spots to detect.
      blur: An :obj:`int`, odd and greater than `1`, defining the size of the
        kernel to use when applying a median blur filter to the image before
        trying to detect spots. Can also be set to :obj:`None`, in which case
        no median blur filter is applied before detecting the spots. Also
        passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.
      update_thresh: If :obj:`True`, the grey level threshold for detecting
        the spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` 
        and not used in this class.
      safe_mode: If :obj:`True`, will stop and raise an exception as soon as
        overlapping spots are detected. Otherwise, will first try to reduce the
        detection window to get rid of overlapping. This argument should be
        used when inconsistency in the results may have critical consequences.
        Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` 
        and not used in this class.
      border: When searching for the new position of a spot, will search in the
        last known bounding box of this spot plus a few additional pixels in
        each direction. This argument sets the number of additional pixels to
        use. It should be greater than the expected "speed" of the spots, in
        pixels / frame. But if set too high, noise or other spots might hinder
        the detection. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` 
        and not used in this class.
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
    and return a :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes`
    object containing all the detected spots.

    Args:
      img: The sub-image on which the spots should be detected.
      y_orig: The y coordinate of the top-left pixel of the sub-image on the
        entire image.
      x_orig: The x coordinate of the top-left pixel of the sub-image on the
        entire image.

    Returns:
      A :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` object
      containing all the detected spots.
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
