# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from itertools import combinations
import logging

from .._global import OptionalModule

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


@dataclass
class Zoom:
  """This class stores the upper and lower limits of the image to display in
  the configuration window.

  It also allows updating them when the user changes the zoom ratio or drags
  the image with the mouse.
  """

  x_low: float = 0.
  x_high: float = 1.
  y_low: float = 0.
  y_high: float = 1.

  def reset(self) -> None:
    """Resets the zoom level to default (no zoom)."""

    self.x_low, self.x_high, self.y_low, self.y_high = 0, 1, 0, 1

  def update_zoom(self, x: float, y: float, ratio: float) -> None:
    """Updates the upper and lower limits of the image when the user scrolls
    with the mousewheel.

    The update is based on the zoom ratio and the position of the mouse on the
    screen.

    Args:
      x: The `x` position of the mouse on the image, as a ratio between `0`
        and `1`.
      y: The `y` position of the mouse on the image, as a ratio between `0`
        and `1`.
      ratio: The zoom ratio to apply. If it is greater than `1` we zoom in,
        otherwise we zoom out.
    """

    prev_x_low, prev_x_high = self.x_low, self.x_high
    prev_y_low, prev_y_high = self.y_low, self.y_high

    # Updating the lower x limit
    self.x_low = max(self.x_low + x * (1 - 1 / ratio), 0.)
    # Updating the upper x limit, making sure it's not out of the image
    if self.x_low + 1 / ratio * (prev_x_high - prev_x_low) > 1.:
      self.x_high = 1.
      self.x_low = 1 - 1 / ratio * (prev_x_high - prev_x_low)
    else:
      self.x_high = self.x_low + 1 / ratio * (prev_x_high - prev_x_low)

    # Updating the lower y limit
    self.y_low = max(self.y_low + y * (1 - 1 / ratio), 0.)
    # Updating the upper y limit, making sure it's not out of the image
    if self.y_low + 1 / ratio * (prev_y_high - prev_y_low) > 1.:
      self.y_high = 1.
      self.y_low = 1 - 1 / ratio * (prev_y_high - prev_y_low)
    else:
      self.y_high = self.y_low + 1 / ratio * (prev_y_high - prev_y_low)

  def update_move(self, delta_x: float, delta_y: float) -> None:
    """Updates the upper and lower limits of the image when the user moves the
    image with a left button click.

    Args:
      delta_x: The `x` displacement to apply to the image, as a ratio of the
        total image width.
      delta_y: The `y` displacement to apply to the image, as a ratio of the
        total image height.
    """

    prev_x_low, prev_x_high = self.x_low, self.x_high
    prev_y_low, prev_y_high = self.y_low, self.y_high

    # Updating the x position
    if delta_x <= 0:
      self.x_low = max(0., prev_x_low + delta_x)
      self.x_high = self.x_low + prev_x_high - prev_x_low
    else:
      self.x_high = min(1., prev_x_high + delta_x)
      self.x_low = self.x_high - prev_x_high + prev_x_low

    # Updating the y position
    if delta_y <= 0:
      self.y_low = max(0., prev_y_low + delta_y)
      self.y_high = self.y_low + prev_y_high - prev_y_low
    else:
      self.y_high = min(1., prev_y_high + delta_y)
      self.y_low = self.y_high - prev_y_high + prev_y_low


@dataclass
class Box:
  """This class represents a box to be drawn on the image of a CameraConfig
  GUI.

  It can be either the box drawn when selecting a region, or the bounding box
  of a previously detected area."""

  x_start: Optional[int] = None
  x_end: Optional[int] = None
  y_start: Optional[int] = None
  y_end: Optional[int] = None

  x_centroid: Optional[float] = None
  y_centroid: Optional[float] = None

  def no_points(self) -> bool:
    """Returns whether all four sides of the box are defined or not."""

    return any(point is None for point in (self.x_start, self.x_end,
                                           self.y_start, self.y_end))

  def reset(self) -> None:
    """Resets the sides to :obj:`None`."""

    self.x_start = None
    self.x_end = None
    self.y_start = None
    self.y_end = None

    self.x_centroid = None
    self.y_centroid = None

  def get_patch(self) -> (int, int, int, int):
    """Returns the information of the box in the patch format, for
    compatibility with other blocks."""

    return (self.y_start, self.x_start, self.y_end - self.y_start,
            self.x_end - self.x_start)

  def sorted(self) -> (int, int, int, int):
    """Returns the four sides values but sorted in the order : min x, max x,
    min y, max y."""

    if self.no_points():
      raise ValueError("Cannot sort, some values are None !")

    x_top = min(self.x_start, self.x_end)
    x_bottom = max(self.x_start, self.x_end)
    y_left = min(self.y_start, self.y_end)
    y_right = max(self.y_start, self.y_end)

    return x_top, x_bottom, y_left, y_right


@dataclass
class Spot_boxes:
  """This class stores up to four instances of :class:`Box`, defining the
  bounding boxes of the spots for video extensometry or the patches for DISVE.

  It can also instantiate the Box object by parsing a list of tuples containing
  enough information.
  """

  spot_1: Optional[Box] = None
  spot_2: Optional[Box] = None
  spot_3: Optional[Box] = None
  spot_4: Optional[Box] = None

  _index = -1

  def __getitem__(self, i: int) -> Optional[Box]:
    if i == 0:
      return self.spot_1
    elif i == 1:
      return self.spot_2
    elif i == 2:
      return self.spot_3
    elif i == 3:
      return self.spot_4
    else:
      raise IndexError

  def __setitem__(self, i: int, value: Optional[Box]) -> None:
    if i == 0:
      self.spot_1 = value
    elif i == 1:
      self.spot_2 = value
    elif i == 2:
      self.spot_3 = value
    elif i == 3:
      self.spot_4 = value
    else:
      raise IndexError

  def __iter__(self):
    self._index = -1
    return self

  def __next__(self) -> Box:
    self._index += 1
    try:
      return self[self._index]
    except IndexError:
      raise StopIteration

  def __len__(self) -> int:
    return len([spot for spot in self if spot is not None])

  def set_spots(self,
                spots: List[Tuple[int, int, int, int]]) -> None:
    """Parses a list of tuples and instantiates the corresponding Box
    objects."""

    for i, spot in enumerate(spots):
      self[i] = Box(x_start=spot[1], x_end=spot[1] + spot[3],
                    y_start=spot[0], y_end=spot[0] + spot[2])

  def empty(self) -> bool:
    """Returns :obj:`True` if all spots are :obj:`None`, else :obj:`False`."""

    return all(spot is None for spot in self)

  def reset(self) -> None:
    """Resets the boxes to :obj:`None`."""

    for i in range(4):
      self[i] = None


class Spot_detector:
  """"""

  def __init__(self,
               logger_name: str,
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

    self._logger = logging.getLogger(f"{logger_name}.{type(self).__name__}")

    if num_spots is not None and num_spots not in range(1, 5):
      raise ValueError("num_spots should be either None, 1, 2, 3 or 4 !")
    self.num_spots = num_spots

    self.spots = Spot_boxes()
    self.x_l0 = None
    self.y_l0 = None

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

  def save_length(self) -> None:
    """Saves the initial length in x and y between the detected spots."""

    # Simply taking the distance between the extrema as the initial length
    if len(self.spots) > 1:
      x_centers = [spot.x_centroid for spot in self.spots if spot is not None]
      y_centers = [spot.y_centroid for spot in self.spots if spot is not None]
      self.x_l0 = max(x_centers) - min(x_centers)
      self.y_l0 = max(y_centers) - min(y_centers)

    # If only one spot detected, setting the initial lengths to 0
    else:
      self.x_l0 = 0
      self.y_l0 = 0

  @staticmethod
  def _overlap_bbox(prop_1, prop_2) -> bool:
    """Determines whether two bboxes are overlapping or not."""

    y_min_1, x_min_1, y_max_1, x_max_1 = prop_1.bbox
    y_min_2, x_min_2, y_max_2, x_max_2 = prop_2.bbox

    return max((min(x_max_1, x_max_2) - max(x_min_1, x_min_2)), 0) * max(
      (min(y_max_1, y_max_2) - max(y_min_1, y_min_2)), 0) > 0
