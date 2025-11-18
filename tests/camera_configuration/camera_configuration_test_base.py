# coding: utf-8

import unittest
from multiprocessing import Queue, queues
import logging
from time import time, sleep
import numpy as np
import sys
import cv2
from platform import system

from . import mock_messagebox

sys.modules['tkinter.messagebox'] = mock_messagebox
from crappy.tool.camera_config.camera_config import CameraConfig
from crappy.tool.camera_config.camera_config_boxes import CameraConfigBoxes
from crappy.tool.camera_config.dic_ve_config import DICVEConfig
from crappy.tool.camera_config.dis_correl_config import DISCorrelConfig
from crappy.tool.camera_config.video_extenso_config import VideoExtensoConfig
from crappy.camera.meta_camera.camera import Camera


class ConfigurationWindowTestBase(unittest.TestCase):
  """Base test class for testing the
  :class:`~crappy.tool.camera_config.CameraConfig` of Crappy.

  Basically implements setup and teardown methods shared by many test classes.

  .. versionadded:: 2.0.8
  """

  def __init__(self,
               *args,
               freq: float | None = None,
               log_level: int | None = None,
               log_queue: queues.Queue | None = None,
               camera: Camera | None = None,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      *args: Positional arguments to pass to the base
        :class:`~unittest.TestCase`.
      freq: The maximum looping frequency the configuration window is allowed
        to loop at.
      log_level: The minimum logging level for the configuration window.
      log_queue: A :obj:`~queues.Queue` for sending the log messages of the
        histogram process to the main logger.
      camera: The :class:`~crappy.camera.Camera` object producing the images
        for the configuration window.
      **kwargs: Keyword arguments to pass to the base
        :class:`~unittest.TestCase`.
    """

    super().__init__(*args, **kwargs)

    self._log_queue: queues.Queue | None = log_queue
    self._log_level: int | None = log_level
    self._freq: float | None = freq
    self._camera: Camera | None = camera

    self._config: (CameraConfig | CameraConfigBoxes | DICVEConfig |
                   DISCorrelConfig | VideoExtensoConfig | None) = None

    self._exit: bool = True

  def setUp(self) -> None:
    """Defines the arguments to pass to the configuration window if not already
    given."""

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
    """Instantiates the configuration window and starts it."""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = True
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def tearDown(self) -> None:
    """Closes the configuration window and the log Queue."""

    self.customTearDown()

    if self._config is not None and self._exit:
      self._config.finish()

    if self._log_queue is not None:
      self._log_queue.close()

  def customTearDown(self) -> None:
    """Meant to be overwritten in subclasses for custom behavior."""

    ...


class FakeTestCameraSimple(Camera):
  """Fake :class:`~crappy.camera.Camera` used for tests, generating a
  grey-level gradient image.

  .. versionadded:: 2.0.8
  """

  def __init__(self, min_val: int = 0, max_val: int = 255) -> None:
    """Initializes the parent class.

    Args:
      min_val: Minimum value in the generated image.
      max_val: Maximum value in the generated image.
    """

    super().__init__()

    self._min = min_val
    self._max = max_val

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a grey-level image containing a gradient from the specified
    minimum to the specified maximum."""

    x, y = np.mgrid[0:240, 0:320]
    ret = (self._min + (x + y) / np.max(x + y) *
           (self._max - self._min)).astype(np.uint8)
    return time(), ret


class FakeTestCameraSpots(Camera):
  """Fake :class:`~crappy.camera.Camera` used for test of the
  video-extensometry configurator, generating a white image with four round
  spots.

  .. versionadded:: 2.0.8
  """

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a white image with four round black spots."""

    ret = np.full((240, 320), 255, dtype=np.uint8)
    ret = cv2.circle(ret, (80, 80), 20, (0,), -1)
    ret = cv2.circle(ret, (80, 160), 20, (0,), -1)
    ret = cv2.circle(ret, (160, 80), 20, (0,), -1)
    ret = cv2.circle(ret, (160, 160), 20, (0,), -1)

    return time(), ret


class FakeTestCameraParams(Camera):
  """Fake :class:`~crappy.camera.Camera` used for testing the parameter
  handling in the configuration interface.

  .. versionadded:: 2.0.8
  """

  def __init__(self) -> None:
    """Initializes the parent class and sets the attributes."""

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
    """Instantiates the four camera parameters to test."""

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
    """Setter for the boolean parameter."""

    self._bool_setter_called = True
    self.settings['bool_setting']._value_no_getter = value

  def _bool_getter(self) -> bool:
    """Getter for the boolean parameter."""

    self._bool_getter_called = True
    return self.settings['bool_setting']._value_no_getter

  def _scale_int_setter(self, value: int) -> None:
    """Setter for the integer parameter."""

    self._scale_int_setter_called = True
    self.settings['scale_int_setting']._value_no_getter = value

  def _scale_int_getter(self) -> int:
    """Getter for the integer parameter."""

    self._scale_int_getter_called = True
    return self.settings['scale_int_setting']._value_no_getter
  
  def _scale_float_setter(self, value: float) -> None:
    """Setter for the float parameter."""

    self._scale_float_setter_called = True
    self.settings['scale_float_setting']._value_no_getter = value

  def _scale_float_getter(self) -> float:
    """Getter for the float parameter."""

    self._scale_float_getter_called = True
    return self.settings['scale_float_setting']._value_no_getter

  def _choice_setter(self, value: str) -> None:
    """Setter for the string parameter."""

    self._choice_setter_called = True
    self.settings['choice_setting']._value_no_getter = value

  def _choice_getter(self) -> str:
    """Getter for the string parameter."""

    self._choice_getter_called = True
    return self.settings['choice_setting']._value_no_getter
