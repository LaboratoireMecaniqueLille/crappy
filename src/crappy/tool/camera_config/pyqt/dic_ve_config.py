# coding: utf-8

"""PyQt6 camera configurator for DICVE."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import DICVEConfig
from ..config_tools import SpotsBoxes
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import CameraScaleSetting
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtDICVEConfig(DICVEConfig, PyQtCameraConfigBoxes):
  """PyQt6 configurator for selecting the four DICVE tracking patches."""

  def __init__(self, camera: Camera, log_queue: MPQueue,
               log_level: int | None, max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patches: SpotsBoxes) -> None:
    """Initializes the window with the tracking patches supplied by the Block.

    Args:
      patches: The four patch boxes to display and update.
    """

    self._patch_size: CameraScaleSetting | None = None
    super().__init__(camera, log_queue, log_level, max_freq, transform)
    self._spots = patches
