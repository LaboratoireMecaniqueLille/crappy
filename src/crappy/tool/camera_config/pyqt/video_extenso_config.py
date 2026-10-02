# coding: utf-8

"""PyQt6 camera configurator for VideoExtenso."""

from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
import numpy as np

from ..base import VideoExtensoConfig
from ..config_tools import SpotsDetector
from ....camera.meta_camera import Camera
from .camera_config_boxes import PyQtCameraConfigBoxes


class PyQtVideoExtensoConfig(VideoExtensoConfig, PyQtCameraConfigBoxes):
  """PyQt6 configurator for initial spot detection and saving L0."""

  def __init__(self, camera: Camera, log_queue: MPQueue,
               log_level: int | None, max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               white_spots: bool, num_spots: int | None, min_area: int,
               blur: int | None, update_thresh: bool, safe_mode: bool,
               border: int) -> None:
    """Initializes the window and the VideoExtenso spot detector.

    The detection arguments are forwarded unchanged to
    :class:`~crappy.tool.camera_config.config_tools.SpotsDetector`.
    """

    super().__init__(camera, log_queue, log_level, max_freq, transform)
    self._detector = SpotsDetector(white_spots=white_spots,
                                   num_spots=num_spots,
                                   min_area=min_area,
                                   blur=blur,
                                   update_thresh=update_thresh,
                                   safe_mode=safe_mode,
                                   border=border)
    self._spots = self._detector.spots
