# coding: utf-8

import unittest
from typing import Optional
from multiprocessing import Queue, queues
import logging
from time import time
import numpy as np
import sys

from . import mock_messagebox
sys.modules['tkinter.messagebox'] = mock_messagebox
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

    self.customSetUp()

  def customSetUp(self) -> None:
    """"""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = True
    self._config.start()

  def tearDown(self) -> None:
    """"""

    if self._config is not None and self._exit:
      self._config.finish()

    if self._log_queue is not None:
      self._log_queue.close()


class FakeTestCameraSimple(Camera):
  """"""

  def __init__(self, min_val: int = 0, max_val: int = 255) -> None:
    """"""

    super().__init__()

    self._min = min_val
    self._max = max_val

  def get_image(self) -> tuple[float, np.ndarray]:
    """"""

    x, y = np.mgrid[0:240, 0:320]
    ret = np.astype(self._min + (x + y) / np.max(x + y) *
                    (self._max - self._min), np.uint8)
    return time(), ret


class FakeTestCameraParams(Camera):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

    self._bool_getter_called: bool = False
    self._bool_setter_called: bool = False
    self._scale_int_getter_called: bool = False
    self._scale_int_setter_called: bool = False
    self._scale_float_getter_called: bool = False
    self._scale_float_setter_called: bool = False
    self._choice_getter_called: bool = False
    self._choice_setter_called: bool = False

    self._scale_int_bounds = (-100, 100, 2)
    self._scale_float_bounds = (-10.0, 10.0, 0.1)
    self._choices = ('choice_1', 'choice_2', 'choice_3')

  def open(self) -> None:
    """"""

    self.add_bool_setting('bool_setting',
                          self._bool_getter,
                          self._bool_setter,
                          True)
    
    self.add_scale_setting('scale_int_setting',
                           self._scale_int_bounds[0],
                           self._scale_int_bounds[1],
                           self._scale_int_getter,
                           self._scale_int_setter,
                           default=0,
                           step=self._scale_int_bounds[2])

    self.add_scale_setting('scale_float_setting',
                           self._scale_float_bounds[0],
                           self._scale_float_bounds[1],
                           self._scale_float_getter,
                           self._scale_float_setter,
                           default=0.,
                           step=self._scale_float_bounds[2])

    self.add_choice_setting('choice_setting',
                            self._choices,
                            self._choice_getter,
                            self._choice_setter,
                            self._choices[0])

    # Left out on purpose
    # self.set_all()

  def _bool_setter(self, value: bool) -> None:
    """"""

    self._bool_setter_called = True
    self.settings['bool_setting']._value_no_getter = value

  def _bool_getter(self) -> bool:
    """"""

    self._bool_getter_called = True
    return self.settings['bool_setting']._value_no_getter

  def _scale_int_setter(self, value: int) -> None:
    """"""

    self._scale_int_setter_called = True
    self.settings['scale_int_setting']._value_no_getter = value

  def _scale_int_getter(self) -> int:
    """"""

    self._scale_int_getter_called = True
    return self.settings['scale_int_setting']._value_no_getter
  
  def _scale_float_setter(self, value: float) -> None:
    """"""

    self._scale_float_setter_called = True
    self.settings['scale_float_setting']._value_no_getter = value

  def _scale_float_getter(self) -> float:
    """"""

    self._scale_float_getter_called = True
    return self.settings['scale_float_setting']._value_no_getter

  def _choice_setter(self, value: str) -> None:
    """"""

    self._choice_setter_called = True
    self.settings['choice_setting']._value_no_getter = value

  def _choice_getter(self) -> str:
    """"""

    self._choice_getter_called = True
    return self.settings['choice_setting']._value_no_getter
