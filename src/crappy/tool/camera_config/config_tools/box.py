# coding: utf-8

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from .overlay_object import Overlay


@dataclass
class Box(Overlay):
  """This class represents a box to be drawn on top of the images of a
  :class:`~crappy.tool.camera_config.CameraConfig` window or
  :class:`~crappy.blocks.camera_processes.Displayer` Process of a
  :class:`~crappy.blocks.Camera` Block.

  It is a child of :class:`~crappy.tool.camera_config.config_tools.Overlay`.

  It can represent either the box drawn when selecting a region, or the
  bounding box of a tracked area.
  
  .. versionadded:: 2.0.0
  """

  x_start: Optional[int] = None
  x_end: Optional[int] = None
  y_start: Optional[int] = None
  y_end: Optional[int] = None

  x_disp: Optional[float] = None
  y_disp: Optional[float] = None

  x_centroid: Optional[float] = None
  y_centroid: Optional[float] = None

  def __post_init__(self) -> None:
    """Needed to have the :obj:`~logging.Logger` of the parent class properly
    initialized."""

    super().__init__()

  def __str__(self) -> str:
    """The string representation of this class, only for debugging."""

    return (f"Box with coordinates ({self.x_start}, {self.y_start}), "
            f"({self.x_end}, {self.y_end})")

  def draw(self, img: np.ndarray) -> None:
    """Draws the Box on top of the given image, and returns the modified image.

    The thickness of the drawn lines adapts to the size of the image, so that
    the lines are always visible even when casting the image to a smaller
    format.
    """

    # First, checking if all points are defined
    if self.no_points():
      self.log(logging.DEBUG, f"Cannot draw {self}, not all points are "
                              f"defined !")

    # Getting the thickness of the lines to draw
    x_top, x_bottom, y_left, y_right = self.sorted()
    max_fact = max(img.shape[0] // 480, img.shape[1] // 640, 1)

    # Drawing the lines on top of the image
    try:
      for line in (line for i in range(max_fact + 1) for line in
                   ((self.y_start + i, slice(x_top, x_bottom)),
                    (self.y_end - i, slice(x_top, x_bottom)),
                    (slice(y_left, y_right), x_top + i),
                    (slice(y_left, y_right), x_bottom - i))):
        img[line] = 255 * int(np.mean(img[line]) < 128)
      self.log(logging.DEBUG, f"Drew {self} on top of the image to display")

    # If anything goes wrong, aborting
    except (Exception,) as exc:
      self._logger.exception("Encountered exception while drawing boxes, "
                             "ignoring", exc_info=exc)

  def update(self, box: Box) -> None:
    """Changes the coordinates of the box to those of another box."""

    self.log(logging.DEBUG, f"Updating {self} to {box}")

    self.x_start = box.x_start
    self.y_start = box.y_start
    self.x_end = box.x_end
    self.y_end = box.y_end

    self.x_disp = box.x_disp
    self.y_disp = box.y_disp

    self.x_centroid = box.x_centroid
    self.y_centroid = box.y_centroid

  def no_points(self) -> bool:
    """Returns whether all four sides of the box are defined or not."""

    return any(point is None for point in (self.x_start, self.x_end,
                                           self.y_start, self.y_end))

  def reset(self) -> None:
    """Resets the sides to :obj:`None`."""

    self.log(logging.DEBUG, f"Resetting {self}")

    self.x_start = None
    self.x_end = None
    self.y_start = None
    self.y_end = None

    self.x_centroid = None
    self.y_centroid = None

  def sorted(self) -> tuple[int, int, int, int]:
    """Returns the four coordinates but sorted in the order : min x, max x,
    min y, max y."""

    if self.no_points():
      self.log(logging.WARNING, f"Trying to sort the Box, but some of its "
                                f"coordinates are undefined !")
      raise ValueError("Cannot sort, some values are None !")

    x_top = min(self.x_start, self.x_end)
    x_bottom = max(self.x_start, self.x_end)
    y_left = min(self.y_start, self.y_end)
    y_right = max(self.y_start, self.y_end)

    self.log(logging.DEBUG, f"Sorted {self}, returning ({x_top}, {x_bottom}, "
                            f"{y_left}, {y_right})")

    return x_top, x_bottom, y_left, y_right
