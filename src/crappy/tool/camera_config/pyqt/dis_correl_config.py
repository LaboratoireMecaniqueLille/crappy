# coding: utf-8

"""PyQt6 camera configurator for DISCorrel."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import DISCorrelConfig
from ..config_tools import Box
from ....camera.meta_camera import Camera
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtDISCorrelConfig(DISCorrelConfig, PyQtCameraConfigBoxes):
  """PyQt6 configurator for selecting a DISCorrel region of interest."""

  def __init__(self, camera: Camera, log_queue: MPQueue,
               log_level: int | None, max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patch: Box) -> None:
    """Initializes the window with the DISCorrel patch supplied by the Block.

    Args:
      patch: The box to display and update as the user changes its selection.
    """

    # The overlay must exist before the first image is drawn
    self._correl_box = patch
    self._draw_correl_box = True
    super().__init__(camera, log_queue, log_level, max_freq, transform)
