# coding: utf-8

import unittest
from typing import Optional
from multiprocessing import Queue, queues
import logging
from time import time
import numpy as np

from crappy.tool.camera_config.camera_config import CameraConfig
from crappy.camera.meta_camera.camera import Camera


class ConfigurationWindowTestBase(unittest.TestCase):
  """"""

  def __init__(self,
               *args,
               freq=None,
               log_level=None,
               log_queue=None,
               camera=None,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class."""

    super().__init__(*args, **kwargs)

    self._log_queue: Optional[queues.Queue] = log_queue
    self._log_level: Optional[int] = log_level
    self._freq: Optional[float] = freq
    self._camera: Optional[Camera] = camera

    self._config: Optional[CameraConfig] = None

    self._exit: bool = True

  def setUp(self) -> None:
    """"""

    if self._log_queue is None:
      self._log_queue = Queue()
    if self._log_level is None:
      self._log_level = logging.CRITICAL
    if self._freq is None:
      self._freq = 30
    if self._camera is None:
      self._camera = Camera()

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._run = False
    self._config.main()
    self._config._run = True

  def tearDown(self) -> None:
    """"""

    if self._config is not None and self._exit:
      self._config.finish()

    if self._log_queue is not None:
      self._log_queue.close()


class FakeTestCameraSimple(Camera):
  """"""

  def get_image(self) -> tuple[float, np.ndarray]:
    """"""

    x, y = np.mgrid[0:320, 0:240]
    ret = np.astype((x + y + 3) / np.max(x + y + 3) * 255, np.uint8)
    return time(), ret
