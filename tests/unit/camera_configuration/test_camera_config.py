# coding: utf-8

"""Headless checks for the camera configurator's backend-neutral core."""

import unittest
import logging
from multiprocessing import current_process
from inspect import signature, Parameter
from unittest.mock import sentinel, Mock

import numpy as np

from crappy.camera.meta_camera.camera_setting import CameraBoolSetting
from crappy.camera.meta_camera import Camera
from crappy._failure import FailureManager
from crappy.tool.camera_config.base import CameraConfig
from crappy.tool.camera_config.factory import create_configurator
from tests.support.assertions import exception_leaves
from tests.support.cameras import DummyCamera
from tests.unit.camera_configuration._fixtures import RecordingCore


class TestCameraConfig(unittest.TestCase):
  """Verify that core state and interactions work without a Tk root."""

  def setUp(self) -> None:
    self.manager = FailureManager('test configuration', 'configuration')

  def test_factory_passes_manager_after_transform(self) -> None:
    """Custom constructors receive the required manager before initialization."""

    class CustomConfig(RecordingCore):
      def __init__(self, camera, log_queue, log_level, max_freq, transform,
                   failure_manager):
        self.received_manager = failure_manager
        super().__init__(camera, log_queue, log_level, max_freq, transform,
                         failure_manager)
        self.initial_manager = self._failure_manager

    for selected in (CustomConfig, {'pyqt': CustomConfig}):
      with self.subTest(selected=selected):
        config = create_configurator(
            selected, DummyCamera(), 'pyqt', sentinel.log_queue, None,
            30, None, self.manager)

        self.assertIs(config.received_manager, self.manager)
        self.assertIs(config._failure_manager, self.manager)
        self.assertIs(config.initial_manager, self.manager)

  def test_factory_rejects_custom_constructor_without_manager(self) -> None:
    """Legacy constructor signatures are no longer adapted by the factory."""

    class CustomConfig(RecordingCore):
      def __init__(self, camera, log_queue, log_level, max_freq, transform):
        super().__init__(camera, log_queue, log_level, max_freq, transform)

    with self.assertRaises(TypeError):
      create_configurator(CustomConfig, DummyCamera(), 'pyqt',
                          sentinel.log_queue, None, 30, None, self.manager)

  def test_manager_is_required_after_transform_in_every_configurator(self):
    """Every concrete constructor has the same mandatory, non-optional slot."""

    from crappy.tool.camera_config import tkinter, pyqt

    classes = [RecordingCore]
    for module, prefix in ((tkinter, 'Tkinter'), (pyqt, 'PyQt')):
      for family in ('CameraConfig', 'CameraConfigBoxes', 'DICVEConfig',
                     'DISCorrelConfig', 'VideoExtensoConfig'):
        classes.append(getattr(module, prefix + family))
    for constructor in (*classes, create_configurator):
      with self.subTest(constructor=constructor):
        # Annotation-only imports are intentionally absent at runtime.
        context = {'Camera': Camera, 'CameraConfig': CameraConfig,
                   'FailureManager': FailureManager, 'np': np}
        parameters = list(signature(constructor, eval_str=True,
                                    locals=context).parameters.values())
        index = next(i for i, item in enumerate(parameters)
                     if item.name == 'failure_manager')
        self.assertEqual(parameters[index - 1].name, 'transform')
        self.assertIs(parameters[index].default, Parameter.empty)
        self.assertIs(parameters[index].annotation, FailureManager)
        self.assertIs(parameters[index].kind, Parameter.POSITIONAL_OR_KEYWORD)

  def test_factory_constructor_failures_use_owner_manager(self) -> None:
    """Initialization and rollback failures enter owner history only once."""

    manager = FailureManager('owner', 'configuration')
    manager._logger = Mock()
    original = ValueError('initialization failed')
    cleanup = OSError('rollback failed')

    class BrokenConfig(RecordingCore):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._failure_manager.collect('Configuration initialization'):
          with self._failure_manager.step('initialize'):
            raise original
          with self._failure_manager.step('rollback'):
            raise cleanup

    with manager.collect('owner', raise_on_exit=False):
      with manager.step('construct configurator'):
        create_configurator(BrokenConfig, DummyCamera(), 'pyqt',
                            sentinel.log_queue, None, 30, None,
                            failure_manager=manager)

    self.assertEqual(tuple(error for _, error in manager.snapshot()),
                     (original, cleanup))
    with self.assertRaises(ExceptionGroup) as caught:
      manager.raise_if_failed('Configuration failures')
    self.assertEqual(exception_leaves(caught.exception), (original, cleanup))

  def test_requires_a_concrete_lifecycle(self) -> None:
    """The base cannot be used before its lifecycle methods are implemented."""

    with self.assertRaises(TypeError):
      CameraConfig(
          DummyCamera(), sentinel.log_queue, None, 30, None, self.manager)

  def test_logging_is_shared_without_a_gui(self) -> None:
    """Normal logs stay local; failures keep the logger supplied by the owner."""

    owner_logger = self.manager._logger = Mock()
    core = RecordingCore(
        DummyCamera(), sentinel.log_queue, None, 30, None, self.manager)
    self.assertIs(self.manager._logger, owner_logger)
    self.assertEqual(core._logger.name,
                     f'{current_process().name}.RecordingCore')
    logger = core._logger = Mock()
    core.log(logging.INFO, 'preview')
    logger.log.assert_called_once_with(logging.INFO, 'preview')
    logger.exception.assert_not_called()
    with self.assertRaises(TypeError):
      core.log(logging.ERROR, 'frame', exc_info=True)
    self.manager.capture_exception(ValueError('frame'))
    owner_logger.error.assert_called_once()
    logger.exception.assert_not_called()

  def test_initializes_settings_and_image_state(self) -> None:
    """The backend-neutral model can be constructed on its own."""

    camera = DummyCamera()
    setting = CameraBoolSetting('enabled', default=True)
    camera.settings['enabled'] = setting

    core = RecordingCore(
        camera, sentinel.log_queue, None, 30, None, self.manager)

    self.assertEqual(core._setting_manager.settings, (setting,))
    self.assertIsNone(core.shape)
    self.assertIsNone(core.dtype)
    self.assertIsNone(core._img)
    self.assertEqual(core._max_freq, 30)

  def test_converts_image_and_updates_reticle_without_widgets(self) -> None:
    """Image conversion and pointer mapping update ordinary state only."""

    core = RecordingCore(
        DummyCamera(), sentinel.log_queue, None, None, None, self.manager)
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
    core = RecordingCore(
        camera, sentinel.log_queue, None, None,
        lambda img: img.astype('uint16') * 2, self.manager)

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

    core = RecordingCore(
        DummyCamera(), sentinel.log_queue, None, None, None, self.manager)
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

    core = RecordingCore(
        DummyCamera(), sentinel.log_queue, None, None, None, self.manager)
    self.assertTrue(core._acquire_image())
    self.assertIsNotNone(core._img)
    self.assertIsNone(core.shape)
    self.assertIsNone(core.dtype)
    self.assertEqual(core._n_loops, 1)

    self.assertFalse(core._acquire_image())
    self.assertEqual(core._n_loops, 1)
