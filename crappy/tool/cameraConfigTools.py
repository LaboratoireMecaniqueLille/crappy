# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Tuple, List
from warnings import warn


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

    warn("The get_patch method will be removed in version 2.0.0",
         DeprecationWarning)

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

  def __post_init__(self) -> None:

    warn("The Spot_boxes class will be renamed to SpotsBoxes in version 2.0.0",
         DeprecationWarning)

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
