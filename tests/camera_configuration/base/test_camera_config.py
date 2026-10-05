# coding: utf-8

"""Headless checks for the camera configurator's backend-neutral core."""

import unittest
import logging
from multiprocessing import current_process
from unittest.mock import patch, sentinel

import numpy as np

from crappy.camera.meta_camera.camera_setting import CameraBoolSetting
from crappy.tool.camera_config.base import CameraConfig
from .._fixtures import DummyCamera
from ._fixtures import RecordingCore


class TestCameraConfig(unittest.TestCase):
  """Verify that core state and interactions work without a Tk root."""

  def test_requires_a_concrete_lifecycle(self) -> None:
    """The base cannot be used before its lifecycle methods are implemented."""

    with self.assertRaises(TypeError):
      CameraConfig(DummyCamera(), sentinel.log_queue, None, 30, None)

  def test_logging_is_shared_without_a_gui(self) -> None:
    """The core records normal messages and explicit exception information."""

    core = RecordingCore(DummyCamera(), sentinel.log_queue, None, 30, None)
    error = ValueError('frame')
    exc_info = (ValueError, error, None)
    with patch('crappy.tool.camera_config.base.camera_config.'
               'logging.getLogger') as get_logger:
      core.log(logging.INFO, 'preview')
      core.log(logging.ERROR, 'frame', exc_info=exc_info)

    get_logger.assert_called_once_with(
        f'{current_process().name}.RecordingCore')
    get_logger.return_value.log.assert_called_once_with(logging.INFO, 'preview')
    get_logger.return_value.exception.assert_called_once_with(
        'frame', exc_info=exc_info)

  def test_initializes_settings_and_image_state(self) -> None:
    """The backend-neutral model can be constructed on its own."""

    camera = DummyCamera()
    setting = CameraBoolSetting('enabled', default=True)
    camera.settings['enabled'] = setting

    core = RecordingCore(camera, sentinel.log_queue, None, 30, None)

    self.assertEqual(core._setting_manager.settings, (setting,))
    self.assertIsNone(core.shape)
    self.assertIsNone(core.dtype)
    self.assertIsNone(core._img)
    self.assertEqual(core._max_freq, 30)

  def test_converts_image_and_updates_reticle_without_widgets(self) -> None:
    """Image conversion and pointer mapping update ordinary state only."""

    core = RecordingCore(DummyCamera(), sentinel.log_queue, None, None, None)
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    core._cast_img(image)
    core._display_geometry.width = 40
    core._display_geometry.height = 40
    core._display_geometry.image_width = 40
    core._display_geometry.image_height = 40

    self.assertTrue(core._point_at(15, 25))
    self.assertEqual((core._display_state.reticle_x,
                      core._display_state.reticle_y,
                      core._display_state.reticle_value), (1, 2, 9))
    self.assertEqual(core._display_state.max_pixel, 15)
    self.assertTrue(core._zoom_at(20, 20, 1))
    self.assertGreater(core._display_state.zoom_percent, 100)

  def test_acquisition_transforms_and_reports_image_format(self) -> None:
    """The frame path and format reporting do not require a GUI loop."""

    camera = DummyCamera()
    camera.frame = ({}, np.ones((2, 3), dtype=np.uint8))
    core = RecordingCore(camera, sentinel.log_queue, None, None,
                         lambda img: img.astype('uint16') * 2)

    self.assertTrue(core._acquire_image())
    self.assertEqual(core.shape, (2, 3))
    self.assertEqual(core.dtype, 'uint16')
    self.assertEqual(core._n_loops, 1)
    assert core._img is not None
    self.assertEqual(core._img.shape, (2, 3))

    camera.frame = None
    self.assertFalse(core._acquire_image())
    self.assertEqual(core._n_loops, 1)

  def test_pan_uses_plain_display_coordinates(self) -> None:
    """Drag movement changes zoom bounds without a GUI event object."""

    core = RecordingCore(DummyCamera(), sentinel.log_queue, None, None, None)
    core._display_geometry.width = 40
    core._display_geometry.height = 40
    core._display_geometry.image_width = 40
    core._display_geometry.image_height = 40
    self.assertTrue(core._zoom_at(20, 20, 1))

    start = core._zoom_values.x_low
    core._begin_pan(20, 20)
    core._pan_to(25, 20)
    self.assertLess(core._zoom_values.x_low, start)

    core._begin_pan(-1, -1)
    stopped = core._zoom_values.x_low
    core._pan_to(30, 20)
    self.assertEqual(core._zoom_values.x_low, stopped)

  def test_placeholder_does_not_report_a_camera_image_format(self) -> None:
    """The no-image preview is display-only and acquired once."""

    core = RecordingCore(DummyCamera(), sentinel.log_queue, None, None, None)
    self.assertTrue(core._acquire_image())
    self.assertIsNotNone(core._img)
    self.assertIsNone(core.shape)
    self.assertIsNone(core.dtype)
    self.assertEqual(core._n_loops, 1)

    self.assertFalse(core._acquire_image())
    self.assertEqual(core._n_loops, 1)
