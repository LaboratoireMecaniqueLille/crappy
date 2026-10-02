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
  """Base Tk configurator for displaying and selecting image-coordinate boxes.

  It extends :class:`~crappy.tool.camera_config.tkinter.camera_config.\
TkinterCameraConfig` with transient box selection and image-array overlays. The
  selection rules live in
  :class:`~crappy.tool.camera_config.base.camera_config_boxes.\
CameraConfigBoxes`. This class binds Tk events and forwards their
  coordinates. A different backend can reuse the same behavior. This class is
  not used directly by a Block.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0
     renamed from *Camera_config_with_boxes* to *CameraConfigBoxes*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
    """Initialize box-selection state and the parent camera configurator.

    Args:
      camera: The :class:`~crappy.camera.meta_camera.camera.Camera` object in
        charge of acquiring the images.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      max_freq: The maximum frequency this window is allowed to loop at. It is
        simply the ``freq`` attribute of the :class:`~crappy.blocks.Camera`
        Block.

        .. versionadded:: 2.0.0
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output.

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
