# coding: utf-8

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass

from .box import Box


@dataclass
class SpotsBoxes:
  """Collection of up to four spot or tracking-patch boxes.

  Iteration visits all four slots, including empty ones. :obj:`len() <len>`
  counts populated slots. Initial lengths are the horizontal and vertical
  distance between extreme centroids, saved independently of the box
  collection.

  Attributes:
    spot_1: First :class:`~crappy.tool.camera_config.config_tools.Box`, or
      :obj:`None` if unused.
    spot_2: Second :class:`~crappy.tool.camera_config.config_tools.Box`, or
      :obj:`None` if unused.
    spot_3: Third :class:`~crappy.tool.camera_config.config_tools.Box`, or
      :obj:`None` if unused.
    spot_4: Fourth :class:`~crappy.tool.camera_config.config_tools.Box`, or
      :obj:`None` if unused.
    x_l0: Initial horizontal centroid distance in pixels, or :obj:`None` if
      unset.
    y_l0: Initial vertical centroid distance in pixels, or :obj:`None` if
      unset.

  .. versionadded:: 2.0.0
  """

  spot_1: Box | None = None
  spot_2: Box | None = None
  spot_3: Box | None = None
  spot_4: Box | None = None

  x_l0: float | None = None
  y_l0: float | None = None

  def __getitem__(self, i: int) -> Box | None:
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

  def __setitem__(self, i: int, value: Box | None) -> None:
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

  def __iter__(self) -> Iterator[Box | None]:
    """Iterates over the four spot slots with an independent iterator."""

    return iter((self.spot_1, self.spot_2, self.spot_3, self.spot_4))

  def __len__(self) -> int:
    return len([spot for spot in self if spot is not None])

  def set_spots(self,
                spots: list[tuple[int, int, int, int]]) -> None:
    """Creates boxes from region coordinates without clearing unused slots.

    Args:
      spots: Up to four (y, x, height, width) tuples in full-image pixels. Call
        :meth:`reset() <crappy.tool.camera_config.config_tools.SpotsBoxes.\
reset>` first when replacing a larger existing collection.
    """

    for i, spot in enumerate(spots):
      self[i] = Box(x_start=spot[1], x_end=spot[1] + spot[3],
                    y_start=spot[0], y_end=spot[0] + spot[2])

  def save_length(self) -> None:
    """Saves horizontal and vertical distance between extreme centroids.

    Missing centroids are calculated from box corners. With at most one
    populated spot, both lengths are zero. Results are stored in x_l0 and y_l0.
    """

    # Calculating the centroids of the spots if not already known
    for spot in self:
      if spot is not None and spot.x_centroid is None:
        min_x, max_x, min_y, max_y = spot.sorted()
        spot.x_centroid = min_x + (max_x - min_x) / 2
        spot.y_centroid = min_y + (max_y - min_y) / 2

    # Simply taking the distance between the extrema as the initial length
    if len(self) > 1:
      x_centers = [spot.x_centroid for spot in self if spot is not None]
      y_centers = [spot.y_centroid for spot in self if spot is not None]
      x_len, y_len = len(x_centers), len(y_centers)
      x_centers = [el for el in x_centers if el is not None]
      y_centers = [el for el in y_centers if el is not None]
      if x_len != len(x_centers):
        raise RuntimeError("One of the spot's x centroid wasn't computed as "
                           "expected")
      if y_len != len(y_centers):
        raise RuntimeError("One of the spot's y centroid wasn't computed as "
                           "expected")
      self.x_l0 = max(x_centers) - min(x_centers)
      self.y_l0 = max(y_centers) - min(y_centers)

    # If only one spot detected, setting the initial lengths to 0
    else:
      self.x_l0 = 0
      self.y_l0 = 0

  def empty(self) -> bool:
    """Returns :obj:`True` if all spots are :obj:`None`, else :obj:`False`."""

    return all(spot is None for spot in self)

  def reset(self) -> None:
    """Clears all four box slots without changing the saved initial lengths."""

    for i in range(4):
      self[i] = None

  def copy(self, use_displacements: bool = False) -> SpotsBoxes:
    """Returns an independent copy of the stored boxes.

    Args:
      use_displacements: If :obj:`True`, adds each
        :class:`~crappy.tool.camera_config.config_tools.Box`'s rounded
        ``x_disp`` and ``y_disp`` to its coordinates. Undefined displacements
        are treated as zero. This is useful for generating overlays when the
        tracked boxes themselves remain at their reference positions.

    Returns:
      A new :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes`
      instance with the same populated slots and initial lengths. Each
      populated slot contains a new
      :class:`~crappy.tool.camera_config.config_tools.Box` created using
      :meth:`~crappy.tool.camera_config.config_tools.Box.__add__`.
    """

    spots = SpotsBoxes(x_l0=self.x_l0, y_l0=self.y_l0)
    for i, spot in enumerate(self):
      if spot is None:
        continue

      x_offset = spot.x_disp if (use_displacements
                                 and spot.x_disp is not None) else 0
      y_offset = spot.y_disp if (use_displacements
                                 and spot.y_disp is not None) else 0
      spots[i] = spot + (round(x_offset), round(y_offset))

    return spots
