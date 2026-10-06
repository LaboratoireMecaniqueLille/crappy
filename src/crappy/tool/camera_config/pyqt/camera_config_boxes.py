# coding: utf-8

"""PyQt6 camera configurator with box selection."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import CameraConfigBoxes
from ....camera.meta_camera import Camera
from .camera_config import PyQtCameraConfig


class PyQtCameraConfigBoxes(CameraConfigBoxes, PyQtCameraConfig):
  """PyQt6 camera configuration with left-button box selection.

  A left-button drag defines a temporary box in full-image pixel coordinates.
  The shared :class:`~crappy.tool.camera_config.base.camera_config_boxes.\
CameraConfigBoxes` hooks determine how a completed box is used. This class
  supplies the backend event handling, not a processing-specific selection
  policy.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               camera: Camera,
               log_queue: MPQueue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
    """Initializes box-selection state and the camera window.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.
    """

    super().__init__(camera, log_queue, log_level, max_freq, transform)
