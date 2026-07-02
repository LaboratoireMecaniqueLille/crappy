# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from crappy._global import DefinitionError
from crappy.camera.meta_camera.camera import Camera
from crappy.camera.meta_camera.camera_setting import (CameraBoolSetting,
                                                      CameraChoiceSetting,
                                                      CameraScaleSetting)
import crappy.camera.meta_camera.camera as camera_module


Camera.classes.pop('UnitTestCamera', None)


class UnitTestCamera(Camera):
  """Concrete Camera used for base class tests."""

  def get_image(self):
    return Camera.get_image(self)


class TestCamera(TestCase):
  """Unit tests for the Camera base class."""

  def tearDown(self) -> None:
    """Removes dynamic classes registered during tests."""

    for name in ('TestRegisteredCamera', 'TestDuplicateCamera'):
      Camera.classes.pop(name, None)

  def test_base_camera_is_abstract(self) -> None:
    """Checks that Camera cannot be instantiated directly."""

    with self.assertRaises(TypeError):
      Camera()

  def test_subclasses_are_registered_by_name(self) -> None:
    """Checks that new Camera subclasses are added to the registry."""

    Camera.classes.pop('TestRegisteredCamera', None)

    cls = type('TestRegisteredCamera',
               (Camera,),
               {'get_image': lambda self: None})

    self.assertIs(Camera.classes['TestRegisteredCamera'], cls)

  def test_duplicate_subclass_names_are_rejected(self) -> None:
    """Checks that two Cameras cannot share a class name."""

    Camera.classes.pop('TestDuplicateCamera', None)
    type('TestDuplicateCamera', (Camera,), {'get_image': lambda self: None})

    with self.assertRaises(DefinitionError):
      type('TestDuplicateCamera',
           (Camera,),
           {'get_image': lambda self: None})

  def test_default_attributes_are_initialized(self) -> None:
    """Checks base instance initialization."""

    camera = UnitTestCamera()

    self.assertEqual(camera.settings, {})
    self.assertEqual(camera.trigger_name, 'trigger')
    self.assertEqual(camera.roi_x_name, 'ROI_x')
    self.assertEqual(camera.roi_y_name, 'ROI_y')
    self.assertEqual(camera.roi_width_name, 'ROI_width')
    self.assertEqual(camera.roi_height_name, 'ROI_height')
    self.assertFalse(camera._soft_roi_set)
    self.assertIn(camera.trigger_name, camera._reserved)
    self.assertIsNone(camera._logger)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and lazy initialization."""

    camera = UnitTestCamera()
    logger_name = f"{current_process().name}.UnitTestCamera"

    with self.assertLogs(logger_name, level='INFO') as captured:
      camera.log(logging.INFO, "message")

    self.assertIs(camera._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_open_calls_set_all(self) -> None:
    """Checks default open implementation."""

    camera = UnitTestCamera()
    camera.add_scale_setting('gain', 0, 10)

    camera.open(gain=8)

    self.assertEqual(camera.gain, 8)
    self.assertTrue(camera.settings['gain'].user_set)

  def test_default_get_image_logs_waits_and_returns_none(self) -> None:
    """Checks fallback image acquisition behavior."""

    camera = UnitTestCamera()
    logs = list()
    camera.log = lambda level, msg: logs.append((level, msg))

    with patch.object(camera_module, 'sleep') as mocked_sleep:
      self.assertIsNone(camera.get_image())

    mocked_sleep.assert_called_once_with(1)
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn('get_img', logs[0][1])

  def test_close_is_noop_by_default(self) -> None:
    """Checks optional close method default behavior."""

    self.assertIsNone(UnitTestCamera().close())

  def test_add_bool_setting(self) -> None:
    """Checks bool setting registration."""

    camera = UnitTestCamera()
    camera.add_bool_setting('enabled', default=False)

    self.assertIsInstance(camera.settings['enabled'], CameraBoolSetting)
    self.assertIs(camera.enabled, False)

  def test_add_scale_setting(self) -> None:
    """Checks scale setting registration."""

    camera = UnitTestCamera()
    camera.add_scale_setting('gain', 0, 10, default=3)

    self.assertIsInstance(camera.settings['gain'], CameraScaleSetting)
    self.assertEqual(camera.gain, 3)

  def test_add_choice_setting(self) -> None:
    """Checks choice setting registration."""

    camera = UnitTestCamera()
    camera.add_choice_setting('mode', ('a', 'b'), default='b')

    self.assertIsInstance(camera.settings['mode'], CameraChoiceSetting)
    self.assertEqual(camera.mode, 'b')

  def test_regular_setting_names_are_validated(self) -> None:
    """Checks setting name validation."""

    camera = UnitTestCamera()

    with self.assertRaises(ValueError):
      camera.add_bool_setting(camera.trigger_name)

    camera.add_bool_setting('enabled')
    with self.assertRaises(ValueError):
      camera.add_scale_setting('enabled', 0, 10)

    with self.assertRaises(ValueError):
      camera.add_choice_setting('open', ('a', 'b'))

  def test_add_trigger_setting(self) -> None:
    """Checks trigger setting registration."""

    camera = UnitTestCamera()
    camera.add_trigger_setting()

    setting = camera.settings[camera.trigger_name]
    self.assertIsInstance(setting, CameraChoiceSetting)
    self.assertEqual(setting.choices, ('Free run',
                                       'Hdw after config',
                                       'Hardware'))
    self.assertEqual(camera.trigger, 'Free run')

  def test_only_one_trigger_setting_is_allowed(self) -> None:
    """Checks trigger setting uniqueness."""

    camera = UnitTestCamera()
    camera.add_trigger_setting()

    with self.assertRaises(ValueError):
      camera.add_trigger_setting()

  def test_add_software_roi_creates_reserved_settings(self) -> None:
    """Checks software ROI setting creation."""

    camera = UnitTestCamera()
    camera.add_software_roi(width=5, height=4)

    self.assertTrue(camera._soft_roi_set)
    self.assertEqual(camera.ROI_x, 0)
    self.assertEqual(camera.ROI_y, 0)
    self.assertEqual(camera.ROI_width, 5)
    self.assertEqual(camera.ROI_height, 4)
    self.assertEqual(camera.settings[camera.roi_x_name].highest, 3)
    self.assertEqual(camera.settings[camera.roi_y_name].highest, 2)

  def test_software_roi_requires_minimum_dimensions(self) -> None:
    """Checks software ROI dimension validation."""

    camera = UnitTestCamera()

    with self.assertRaises(ValueError):
      camera.add_software_roi(width=2, height=4)
    with self.assertRaises(ValueError):
      camera.add_software_roi(width=4, height=2)

  def test_only_one_software_roi_can_be_added(self) -> None:
    """Checks software ROI uniqueness."""

    camera = UnitTestCamera()
    camera.add_software_roi(width=5, height=4)

    with self.assertRaises(ValueError):
      camera.add_software_roi(width=5, height=4)

  def test_reload_software_roi_without_roi_logs_warning(self) -> None:
    """Checks reload behavior before ROI settings exist."""

    camera = UnitTestCamera()
    logs = list()
    camera.log = lambda level, msg: logs.append((level, msg))

    camera.reload_software_roi(width=5, height=4)

    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn('Cannot reload', logs[0][1])

  def test_reload_software_roi_updates_limits_and_values(self) -> None:
    """Checks software ROI reload behavior."""

    camera = UnitTestCamera()
    camera.add_software_roi(width=5, height=4)
    camera.ROI_x = 1
    camera.ROI_y = 1

    camera.reload_software_roi(width=7, height=6)

    self.assertEqual(camera.ROI_x, 0)
    self.assertEqual(camera.ROI_y, 0)
    self.assertEqual(camera.ROI_width, 7)
    self.assertEqual(camera.ROI_height, 6)
    self.assertEqual(camera.settings[camera.roi_x_name].highest, 6)
    self.assertEqual(camera.settings[camera.roi_y_name].highest, 5)

  def test_apply_soft_roi_without_roi_returns_input_image(self) -> None:
    """Checks ROI passthrough before settings exist."""

    camera = UnitTestCamera()
    img = np.arange(20).reshape(4, 5)

    self.assertIs(camera.apply_soft_roi(img), img)

  def test_apply_soft_roi_crops_image(self) -> None:
    """Checks software ROI cropping."""

    camera = UnitTestCamera()
    camera.add_software_roi(width=5, height=4)
    camera.ROI_x = 1
    camera.ROI_y = 1
    camera.ROI_width = 2
    camera.ROI_height = 2
    img = np.arange(20).reshape(4, 5)

    np.testing.assert_array_equal(camera.apply_soft_roi(img),
                                  img[1:3, 1:3])

  def test_set_all_rejects_unexpected_kwargs(self) -> None:
    """Checks set_all kwarg validation."""

    camera = UnitTestCamera()

    with self.assertRaises(ValueError):
      camera.set_all(missing=1)

  def test_set_all_sets_defaults_and_user_values(self) -> None:
    """Checks applying all setting values."""

    camera = UnitTestCamera()
    camera.add_bool_setting('enabled', default=True)
    camera.add_scale_setting('gain', 0, 10, default=3)

    camera.set_all(gain=8)

    self.assertIs(camera.enabled, True)
    self.assertEqual(camera.gain, 8)
    self.assertFalse(camera.settings['enabled'].user_set)
    self.assertTrue(camera.settings['gain'].user_set)

  def test_setting_values_can_be_read_and_written_as_attributes(self) -> None:
    """Checks Camera __getattr__ and __setattr__ setting shortcuts."""

    camera = UnitTestCamera()
    camera.add_scale_setting('gain', 0, 10, default=3)

    camera.gain = 9

    self.assertEqual(camera.gain, 9)

  def test_unknown_attribute_raises_attribute_error(self) -> None:
    """Checks unknown attribute behavior."""

    with self.assertRaises(AttributeError):
      UnitTestCamera().unknown
