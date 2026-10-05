# coding: utf-8

"""Backend-independent camera configuration with box selection and drawing."""

import logging
from typing import Any
import numpy as np

from .camera_config import CameraConfig
from ..config_tools.box import Box
from ..config_tools.spots_boxes import SpotsBoxes


class CameraConfigBoxes(CameraConfig):
  """Abstract camera configuration with box-selection and drawing hooks.

  Selection gestures use display coordinates and are converted to full-image
  pixels through the shared geometry and zoom. Temporary boxes and overlays
  affect preview pixels only. Subclasses define what a completed box means
  and implement the lifecycle through an implemented backend.

  .. versionadded:: 2.1.0
  """

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes transient selection state and the shared configuration.

    Args:
      *args: Positional arguments forwarded through cooperative initialization.
      **kwargs: Keyword arguments forwarded through cooperative initialization.
    """

    self._spots: SpotsBoxes = SpotsBoxes()
    self._select_box: Box = Box()

    super().__init__(*args, **kwargs)

  def _start_box_at(self, x: int, y: int) -> None:
    """Start a selection at display coordinates if the press hits the image.

    Any unfinished selection is canceled first, including for an outside
    press. Specialized behavior can react through ``_on_selection_start``.
    """

    # A new press cancels a previous drag, including an outside press
    self._select_box.reset()
    self._on_selection_end()

    # Only continue if there is an image and the click is inside
    if self._img is None or not self._is_on_image(x, y):
      return

    self.log(logging.DEBUG, "Starting the selection box")
    (self._select_box.x_start,
     self._select_box.y_start) = self._selection_to_pixel(x, y)
    self._on_selection_start()

  def _extend_box_to(self, x: int, y: int) -> None:
    """Update the selection endpoint from a drag on the displayed image."""

    # Only continue if there is an image and the click is inside and the
    # selection box was initialized
    if (self._select_box.x_start is None or
        self._select_box.y_start is None or
        self._img is None or not self._is_on_image(x, y)):
      return

    (self._select_box.x_end,
     self._select_box.y_end) = self._selection_to_pixel(x, y)
    self._on_selection_drag()

  def _complete_box_selection(self) -> None:
    """Accept a nonflat box, then clear the transient box even on failure."""

    try:
      if self._img is not None and self._selection_is_valid():
        self._on_selection_complete(self._select_box)
    finally:
      self._select_box.reset()
      self._on_selection_end()

  def _selection_to_pixel(self, x: int, y: int) -> tuple[int, int]:
    """Convert display coordinates to source pixels using geometry and zoom."""

    if self._img is None:
      return 0, 0

    height, width, *_ = self._img.shape
    return self._display_geometry.to_pixel(x, y, width, height,
                                           self._zoom_values)

  def _selection_is_valid(self) -> bool:
    """Whether all four sides exist and span a nonzero area."""

    if self._select_box.no_points():
      return False

    min_x, max_x, min_y, max_y = self._select_box.sorted()
    return min_x < max_x and min_y < max_y

  def _on_selection_start(self) -> None:
    """Reacts to a valid press that begins a selection.

    Override this hook to prepare a specialized selection. The default does
    nothing.
    """

    ...

  def _on_selection_drag(self) -> None:
    """Reacts when dragging changes the transient selection box.

    Override this hook to update derived selections. The default does nothing.
    """

    ...

  def _on_selection_complete(self, box: Box) -> None:
    """Consumes a valid selection before the transient box is cleared.

    The default does nothing.

    Args:
      box: Temporary box in full-image pixel coordinates. Copy its values into
        persistent state, as the same box is reset after this hook returns.
    """

    ...

  def _on_selection_end(self) -> None:
    """Reacts when a selection completes or is canceled.

    This also runs before a new press clears an unfinished selection. Override
    it to restore drawing state. The default does nothing.
    """

    ...

  def _handle_box_outside_img(self, box: Box) -> None:
    """Handles a selection that no longer fits the current image.

    Override this hook to discard or reset invalid selections. The default does
    nothing.

    Args:
      box: :class:`~crappy.tool.camera_config.config_tools.Box` whose
        coordinates fall outside the current preview image.
    """

    ...

  def _draw_box(self, box: Box) -> None:
    """Draw a box into preview pixels with visible geometry-scaled edges."""

    # Only continue if there's a box, an image, and the box fits the image
    if self._img is None or box.no_points():
      return
    min_x, max_x, min_y, max_y = box.sorted()
    height, width, *_ = self._img.shape
    if not (0 <= min_x < max_x <= width and
            0 <= min_y < max_y <= height):
      self._handle_box_outside_img(box)
      return

    geometry = self._display_geometry
    thickness = max(height // max(geometry.height, 1),
                    width // max(geometry.width, 1), 1)
    thickness = min(thickness, max_x - min_x, max_y - min_y)

    # Adjust box color based on the average underlying image color
    for region in (self._img[min_y:min_y + thickness, min_x:max_x],
                   self._img[max_y - thickness:max_y, min_x:max_x],
                   self._img[min_y:max_y, min_x:min_x + thickness],
                   self._img[min_y:max_y, max_x - thickness:max_x]):
      region[:] = 255 * int(np.mean(region) < 128)

  def _draw_spots(self) -> None:
    """Draw all current spots and stop if an invalid spot resets the group."""

    if self._img is None:
      return

    for spot in self._spots:
      if spot is not None:
        self._draw_box(spot)
        # Could potentially be reset while drawing
        if self._spots.empty():
          return
