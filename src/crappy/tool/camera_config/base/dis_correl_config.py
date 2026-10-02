# coding: utf-8

"""Backend-independent configuration of a DISCorrel region of interest."""

from .camera_config_boxes import CameraConfigBoxes
from ..config_tools.box import Box


class DISCorrelConfig(CameraConfigBoxes):
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
