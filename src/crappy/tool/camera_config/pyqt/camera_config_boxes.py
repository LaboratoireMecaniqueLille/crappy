# coding: utf-8

"""PyQt6 camera configurator with box selection."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import CameraConfigBoxes
from ....camera.meta_camera import Camera
from .camera_config import PyQtCameraConfig


class PyQtCameraConfigBoxes(CameraConfigBoxes, PyQtCameraConfig):
  """PyQt6 configurator allowing the user to draw boxes on the image.

  Left-dragging selects a box. The Block-specific behaviors decide how that
  box is used when the selection is completed.
  """

  def __init__(self, camera: Camera, log_queue: MPQueue,
               log_level: int | None, max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
    """Initializes the standard window with box-selection gestures."""

    super().__init__(camera, log_queue, log_level, max_freq, transform)
