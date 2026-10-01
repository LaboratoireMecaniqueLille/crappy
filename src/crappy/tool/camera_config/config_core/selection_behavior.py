# coding: utf-8

"""Toolkit-independent selection and specialized camera configuration logic.

The host supplies image arrays, geometry, zoom, hit testing, and logging. A GUI
backend only translates its pointer events into the coordinate methods below.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..config_tools.box import Box
from ..config_tools.spots_boxes import SpotsBoxes
from ..config_tools.spots_detector import SpotsDetector
from ..config_tools.zoom import Zoom
from .display_state import DisplayGeometry
from ....camera.meta_camera.camera_setting import (CameraSetting,
                                                   CameraScaleSetting)


@dataclass(frozen=True)
class ConfigAction:
  """A semantic action that a GUI backend can present as a button.

  Attributes:
    id: Stable key used to find the action's backend control.
    label: Text displayed on the control.
    callback: Operation to run when the action is activated.
  """

  id: str
  label: str
  callback: Callable[(), None]


class BoxSelectionBehavior:
  """Selection lifecycle and image-array overlays shared by GUI backends.

  The host supplies ``_img``, ``_original_img``, ``_display_geometry``,
  ``_zoom_values``, ``_is_on_image()``, and ``log()``. The attribute
  declarations below describe that contract without creating GUI objects or
  replacing the host's methods at runtime.
  """

  _img: np.ndarray | None
  _original_img: np.ndarray | None
  _display_geometry: DisplayGeometry
  _zoom_values: Zoom
  _spots: SpotsBoxes
  _select_box: Box
  _is_on_image: Callable[[int, int], bool]
  log: Callable[[int, str], None]

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Create transient selection state, then initialize the backend host."""

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
    """Hook called after a valid press begins a selection.

    Meant to be overridden in children classes.
    """

    ...

  def _on_selection_drag(self) -> None:
    """Hook called after a drag changes the transient box.

    Meant to be overridden in children classes.
    """

    ...

  def _on_selection_complete(self, box: Box) -> None:
    """Hook called with a valid box before its coordinates are cleared.

    Meant to be overridden in children classes.
    """

    ...

  def _on_selection_end(self) -> None:
    """Hook called when a selection ends or is canceled.

    Meant to be overridden in children classes.
    """

    ...

  def _handle_box_outside_img(self, box: Box) -> None:
    """Hook for invalidating a box after the image dimensions change.

    Meant to be overridden in children classes.
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


class DISCorrelBehavior(BoxSelectionBehavior):
  """Manage the correlation ROI without depending on a GUI toolkit."""

  _correl_box: Box
  _draw_correl_box: bool

  def _on_selection_start(self) -> None:
    """Hide the old ROI while the user draws a replacement."""

    self._draw_correl_box = False

  def _on_selection_complete(self, box: Box) -> None:
    """Copy a valid selection into the caller-owned ROI box."""

    self._correl_box.update(box)

  def _on_selection_end(self) -> None:
    """Show the old or newly committed ROI after the gesture."""

    self._draw_correl_box = True

  def _draw_overlay(self) -> None:
    """Draw the stored ROI unless a new one is being selected."""

    if self._draw_correl_box:
      self._draw_box(self._correl_box)
    self._draw_box(self._select_box)

  def _handle_box_outside_img(self, box: Box) -> None:
    """Reset an ROI that no longer fits the image, or cancel a drag."""

    if box is self._correl_box:
      self._correl_box.reset()
    else:
      self._select_box.reset()
      self._on_selection_end()

  def _validate_close(self) -> str | None:
    """Require an ROI before allowing the configuration window to close."""

    if self._correl_box.no_points():
      return ("Please select a ROI before exiting the config window!\n"
              "Or hit CTRL+C to exit Crappy")
    return None

  def get_config(self) -> tuple[Box]:
    """Export the same ROI box object supplied to the configurator."""

    return self._correl_box,

  @property
  def box(self) -> Box:
    """The caller-owned correlation region of interest."""

    return self._correl_box


class DICVEBehavior(BoxSelectionBehavior):
  """Construct and validate four tracking patches without GUI dependencies."""

  _patch_size: CameraScaleSetting | None

  def _create_local_settings(self) -> tuple[CameraSetting, ...]:
    """Create the patch-size setting before camera settings are presented."""

    patch_size = CameraScaleSetting("Patch size (px)", 2, 1024,
                                    default=128)
    self._patch_size = patch_size
    return patch_size,

  def _on_selection_drag(self) -> None:
    """Position four patches once both selection spans reach three sizes."""

    if not self._selection_is_valid() or self._patch_size is None:
      return

    min_x, max_x, min_y, max_y = self._select_box.sorted()
    # Using ints here, although users can provide floats
    size = round(self._patch_size.value)
    # Need at least three times the patch size to draw anything
    if max_x - min_x < 3 * size or max_y - min_y < 3 * size:
      return

    self._spots.spot_1 = Box(min_x, min_x + size,
                             (min_y + max_y - size) // 2,
                             (min_y + max_y + size) // 2)
    self._spots.spot_2 = Box(max_x - size, max_x,
                             (min_y + max_y - size) // 2,
                             (min_y + max_y + size) // 2)
    self._spots.spot_3 = Box((min_x + max_x - size) // 2,
                             (min_x + max_x + size) // 2,
                             min_y, min_y + size)
    self._spots.spot_4 = Box((min_x + max_x - size) // 2,
                             (min_x + max_x + size) // 2,
                             max_y - size, max_y)

  def _draw_overlay(self) -> None:
    """Draw the current patches over the preview image."""

    self._draw_spots()

  def _handle_box_outside_img(self, box: Box) -> None:
    """Discard all patches when one no longer fits the image."""

    self.log(logging.WARNING, f"The patch {box} is outside the image, "
                              "resetting the patches")
    self._spots.reset()

  def _validate_close(self) -> str | None:
    """Require at least one patch before allowing close."""

    if self._spots.empty():
      return ("Please select patches before exiting the config window!\n"
              "Or hit CTRL+C to exit Crappy")
    return None

  def _on_valid_close(self) -> None:
    """Save the initial patch separation after close validation succeeds."""

    self._spots.save_length()
    self.log(logging.INFO,
             f"Successfully saved L0. L0 x: {self._spots.x_l0}, "
             f"L0 y: {self._spots.y_l0}")

  def get_config(self) -> tuple[SpotsBoxes]:
    """Export the caller-owned collection of tracking patches."""

    return self._spots,


class VideoExtensoBehavior(BoxSelectionBehavior):
  """Detect spots from a crop and manage their initial lengths without Tk."""

  _detector: SpotsDetector

  def _extra_actions(self) -> tuple[ConfigAction, ...]:
    """Offer Save L0 independently of the backend's Apply Settings action."""

    return (ConfigAction("save_l0", "Save L0", self._save_l0),)

  def _on_selection_complete(self, box: Box) -> None:
    """Detect spots in the selected crop of the unmodified source image."""

    if self._original_img is None:
      return

    min_x, max_x, min_y, max_y = box.sorted()
    try:
      self._detector.detect_spots(self._original_img[min_y:max_y, min_x:max_x],
                                  min_y, min_x)
    except IndexError:
      self._detector.spots.reset()

  def _save_l0(self) -> None:
    """Save initial spot separation when detection has found spots."""

    if self._detector.spots.empty():
      self.log(logging.WARNING, "Cannot save L0, there are no spots!")
      return

    self._detector.spots.save_length()
    self.log(logging.INFO,
             f"Successfully saved L0! L0 x: {self._detector.spots.x_l0}, "
             f"L0 y: {self._detector.spots.y_l0}")

  def _draw_overlay(self) -> None:
    """Draw the selected crop and the detected spots on preview pixels."""

    self._draw_box(self._select_box)
    self._draw_spots()

  def _handle_box_outside_img(self, box: Box) -> None:
    """Discard spots that no longer fit the current image."""

    self._spots.reset()

  def _validate_close(self) -> str | None:
    """Require detected spots before allowing the window to close."""

    if self._detector.spots.empty():
      return ("Please select spots before exiting the config window!\n"
              "Or hit CTRL+C to exit Crappy")
    return None

  def _on_valid_close(self) -> None:
    """Save L0 on close if the user has not saved it already."""

    if self._detector.spots.x_l0 is None or self._detector.spots.y_l0 is None:
      self._save_l0()

  def get_config(self) -> tuple[SpotsBoxes, int]:
    """Export detected spot boxes and threshold for the processing Block."""

    return self._detector.spots, self._detector.thresh
