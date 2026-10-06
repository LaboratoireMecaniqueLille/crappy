# coding: utf-8

from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np

from .overlay_object import Overlay


@dataclass
class Box(Overlay):
  """Image-coordinate rectangle with optional tracking data.

  Used for selection regions, tracking patches, and display overlays.
  Configuration windows draw it with preview-scaled line thickness. Its own
  :meth:`draw() <crappy.tool.camera_config.config_tools.Box.draw>` method
  supports runtime display overlays.

  Attributes:
    x_start: First horizontal corner coordinate, or :obj:`None` if unset.
    x_end: Opposite horizontal corner coordinate, or :obj:`None` if unset.
    y_start: First vertical corner coordinate, or :obj:`None` if unset.
    y_end: Opposite vertical corner coordinate, or :obj:`None` if unset.
    x_disp: Optional horizontal tracking displacement.
    y_disp: Optional vertical tracking displacement.
    x_centroid: Optional horizontal center coordinate.
    y_centroid: Optional vertical center coordinate.

  .. versionadded:: 2.0.0
  """

  x_start: int | None = None
  x_end: int | None = None
  y_start: int | None = None
  y_end: int | None = None

  x_disp: float | None = None
  y_disp: float | None = None

  x_centroid: float | None = None
  y_centroid: float | None = None

  def __post_init__(self) -> None:
    """Needed to have the :obj:`~logging.Logger` of the parent class properly
    initialized."""

    super().__init__()

  def __str__(self) -> str:
    """The string representation of this class, only for debugging."""

    return (f"Box with coordinates ({self.x_start}, {self.y_start}), "
            f"({self.x_end}, {self.y_end})")

  def __add__(self, other: tuple[int, int]) -> Box:
    """Returns a box shifted by an integer pixel offset.

    Args:
      other: :obj:`tuple` containing horizontal and vertical integer offsets.

    Returns:
      A new :class:`~crappy.tool.camera_config.config_tools.Box` with
      shifted corners, defined displacements, and centroids. If any corner is
      unset, returns the original
      :class:`~crappy.tool.camera_config.config_tools.Box` unchanged.

    Raises:
      TypeError: If other is not a :obj:`tuple`.
      ValueError: If the :obj:`tuple` does not contain two integers.
    """

    if not isinstance(other, tuple):
      raise TypeError("Can only add a Box with a tuple of two integers")
    if len(other) != 2 or not all(isinstance(el, int) for el in other):
      raise ValueError("Can only add a Box with a tuple of two integers")

    if self.no_points():
      return self

    x_offset, y_offset = other
    return Box(self.x_start + x_offset,
               self.x_end + x_offset,
               self.y_start + y_offset,
               self.y_end + y_offset,
               self.x_disp + x_offset if self.x_disp is not None else None,
               self.y_disp + y_offset if self.y_disp is not None else None,
               self.x_centroid + x_offset if self.x_centroid is not None
               else None,
               self.y_centroid + y_offset if self.y_centroid is not None
               else None)

  def draw(self, img: np.ndarray) -> None:
    """Draws visible rectangle edges into an image in place.

    Edge thickness adapts to image dimensions, and brightness contrasts with
    the underlying pixels.

    Args:
      img: Display image array to modify.

    Raises:
      ValueError: If corner coordinates are incomplete.
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
                    (self.y_end - i - 1, slice(x_top, x_bottom)),
                    (slice(y_left, y_right), x_top + i),
                    (slice(y_left, y_right), x_bottom - i - 1))):
        img[line] = 255 * int(np.mean(img[line]) < 128)
      self.log(logging.DEBUG, f"Drew {self} on top of the image to display")

    # If anything goes wrong, aborting
    except (Exception,) as exc:
      self._logger.exception("Encountered exception while drawing boxes, "
                             "ignoring", exc_info=exc)

  def update(self, box: Box) -> None:
    """Copies another box's coordinates and tracking fields into this box.

    Args:
      box: Source :class:`~crappy.tool.camera_config.config_tools.Box`. This
        method copies field values without replacing this object, so references
        held by the owning :class:`~crappy.blocks.meta_block.block.Block`
        remain valid.
    """

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
    """Whether at least one of the four corner coordinates is unset."""

    return any(point is None for point in (self.x_start, self.x_end,
                                           self.y_start, self.y_end))

  def reset(self) -> None:
    """Clears corner coordinates and centroids, retaining displacements."""

    self.log(logging.DEBUG, f"Resetting {self}")

    self.x_start = None
    self.x_end = None
    self.y_start = None
    self.y_end = None

    self.x_centroid = None
    self.y_centroid = None

  def sorted(self) -> tuple[int, int, int, int]:
    """Returns ordered corner coordinates.

    Returns:
      :obj:`tuple` of minimum x, maximum x, minimum y, and maximum y
      coordinates.

    Raises:
      ValueError: If any corner coordinate is unset.
    """

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
