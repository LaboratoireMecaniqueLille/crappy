# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Tuple


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

  x_disp: Optional[float] = None
  y_disp: Optional[float] = None

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

  def get_patch(self) -> Tuple[int, int, int, int]:
    """Returns the information of the box in the patch format, for
    compatibility with other blocks."""

    return (self.y_start, self.x_start, self.y_end - self.y_start,
            self.x_end - self.x_start)

  def sorted(self) -> Tuple[int, int, int, int]:
    """Returns the four sides values but sorted in the order : min x, max x,
    min y, max y."""

    if self.no_points():
      raise ValueError("Cannot sort, some values are None !")

    x_top = min(self.x_start, self.x_end)
    x_bottom = max(self.x_start, self.x_end)
    y_left = min(self.y_start, self.y_end)
    y_right = max(self.y_start, self.y_end)

    return x_top, x_bottom, y_left, y_right
