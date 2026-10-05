# coding: utf-8

"""Tk adapter for reusable box-selection behavior."""

import tkinter as tk
from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config import TkinterCameraConfig
from ..base import CameraConfigBoxes
from ....camera.meta_camera import Camera


class TkinterCameraConfigBoxes(CameraConfigBoxes, TkinterCameraConfig):
  """Tkinter camera configuration with left-button box selection.

  A left-button drag defines a temporary box in full-image pixel coordinates.
  The shared :class:`~crappy.tool.camera_config.base.camera_config_boxes.\
CameraConfigBoxes` hooks determine how a completed box is used. This class
  supplies the backend event handling, not a processing-specific selection
  policy.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0
     renamed from *Camera_config_with_boxes* to *CameraConfigBoxes*
  .. versionchanged:: 2.1.0 renamed from *CameraConfigBoxes* to
     *TkinterCameraConfigBoxes*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
    """Initializes box-selection state and the camera window.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.

        .. versionadded:: 2.0.0
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.

        .. versionadded:: 2.0.0
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.

        .. versionadded:: 2.0.0
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.

        .. versionadded:: 2.1.0
    """

    super().__init__(camera, log_queue, log_level, max_freq, transform)

  def _set_bindings(self) -> None:
    """Bind Tk left-button events to the shared selection lifecycle."""

    super()._set_bindings()
    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _start_box(self, event: tk.Event) -> None:
    """Translate a Tk button press into display coordinates."""

    self._start_box_at(event.x, event.y)

  def _extend_box(self, event: tk.Event) -> None:
    """Translate a Tk button drag into display coordinates."""

    self._extend_box_to(event.x, event.y)

  def _stop_box(self, _: tk.Event) -> None:
    """Complete the current selection after a Tk button release."""

    self._complete_box_selection()
