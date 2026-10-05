# coding: utf-8

"""Backend-independent spot detection and configuration for VideoExtenso."""

import logging

from .camera_config import ConfigAction
from .camera_config_boxes import CameraConfigBoxes
from ..config_tools.box import Box
from ..config_tools.spots_boxes import SpotsBoxes
from ..config_tools.spots_detector import SpotsDetector


class VideoExtensoConfig(CameraConfigBoxes):
  """Detect spots from a crop and manage their initial lengths without a
  GUI."""

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
      self.log(logging.WARNING, "Cannot save initial distance, no spots "
                                "are selected")
      return

    self._detector.spots.save_length()
    self.log(logging.INFO,
             f"Saved initial spot distance (px): "
             f"x={self._detector.spots.x_l0}, y={self._detector.spots.y_l0}")

  def _draw_overlay(self) -> None:
    """Draw the selected crop and the detected spots on preview pixels."""

    self._draw_box(self._select_box)
    self._draw_spots()

  def _handle_box_outside_img(self, box: Box) -> None:
    """Discard spots that no longer fit the current image."""

    self.log(logging.WARNING, "A detected spot no longer fits the image, "
                              "resetting the spots")
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
