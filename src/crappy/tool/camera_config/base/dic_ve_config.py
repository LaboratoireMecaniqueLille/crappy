# coding: utf-8

"""Backend-independent configuration of :class:`~crappy.blocks.DICVE` tracking
patches."""

import logging

from .camera_config_boxes import CameraConfigBoxes
from ..config_tools.box import Box
from ..config_tools.spots_boxes import SpotsBoxes
from ....camera.meta_camera.camera_setting import (CameraSetting,
                                                   CameraScaleSetting)


class DICVEConfig(CameraConfigBoxes):
  """Abstract configuration of digital image correlation tracking patches.

  A left-button drag places four patches around a rectangle once both spans
  reach three times the local Patch size setting. Closing requires a non-empty
  patch collection and saves its initial horizontal and vertical distance.
  Implemented backends supply the caller-owned patch collection and GUI
  lifecycle.

  .. versionadded:: 2.1.0
  """

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
             f"Saved initial patch distance (px): x={self._spots.x_l0}, "
             f"y={self._spots.y_l0}")

  def get_config(self) -> tuple[SpotsBoxes]:
    """Returns the selected patches and their saved initial distance.

    Returns:
      One-element :obj:`tuple` containing the same
      :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` collection
      supplied to the configuration window.
      :class:`~crappy.tool.camera_config.config_tools.Box` coordinates are
      full-image pixel coordinates.
    """

    return self._spots,
