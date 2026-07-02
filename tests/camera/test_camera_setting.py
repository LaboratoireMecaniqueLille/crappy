# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase

from crappy.camera.meta_camera.camera_setting import (CameraBoolSetting,
                                                      CameraChoiceSetting,
                                                      CameraScaleSetting,
                                                      CameraSetting)


class FakeTkVar:
  """Tiny stand-in for Tk variables used by camera settings."""

  def __init__(self) -> None:
    self.calls = list()

  def set(self, value) -> None:
    self.calls.append(value)


class FakeButton:
  """Tiny stand-in for Tk radio buttons used by choice settings."""

  def __init__(self) -> None:
    self.configs = list()

  def configure(self, **kwargs) -> None:
    self.configs.append(kwargs)


class FakeScale:
  """Tiny stand-in for Tk scales used by scale settings."""

  def __init__(self) -> None:
    self.configs = list()

  def configure(self, **kwargs) -> None:
    self.configs.append(kwargs)


class TestCameraSetting(TestCase):
  """Unit tests for the CameraSetting base class."""

  def test_initialization_stores_common_attributes(self) -> None:
    """Checks common setting attributes."""

    setting = CameraSetting('gain', None, None, 5)

    self.assertEqual(setting.name, 'gain')
    self.assertEqual(setting.default, 5)
    self.assertIs(setting.type, int)
    self.assertFalse(setting.was_set)
    self.assertFalse(setting.user_set)
    self.assertIsNone(setting.tk_var)
    self.assertIsNone(setting.tk_obj)
    self.assertEqual(setting.value, 5)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and lazy initialization."""

    setting = CameraSetting('gain', None, None, 5)
    logger_name = f"{current_process().name}.CameraSetting"

    with self.assertLogs(logger_name, level='INFO') as captured:
      setting.log(logging.INFO, "message")

    self.assertIs(setting._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_value_without_getter_is_stored_locally(self) -> None:
    """Checks setting values without getter or setter."""

    setting = CameraSetting('gain', None, None, 5)

    setting.value = 7

    self.assertEqual(setting.value, 7)
    self.assertTrue(setting.was_set)

  def test_value_with_getter_and_setter_uses_callbacks(self) -> None:
    """Checks getter and setter integration."""

    state = {'value': 5}

    def getter():
      return state['value']

    def setter(value):
      state['value'] = value

    setting = CameraSetting('gain', getter, setter, 5)

    setting.value = 8

    self.assertEqual(setting.value, 8)
    self.assertEqual(state['value'], 8)
    self.assertTrue(setting.was_set)

  def test_value_logs_warning_when_getter_disagrees(self) -> None:
    """Checks mismatch warning after setting through a callback."""

    logs = list()
    setting = CameraSetting('gain', lambda: 3, lambda _: None, 5)
    setting.log = lambda level, msg: logs.append((level, msg))

    setting.value = 8

    self.assertEqual(logs[-1][0], logging.WARNING)
    self.assertIn('Could not set gain to 8', logs[-1][1])

  def test_value_updates_tk_variable(self) -> None:
    """Checks GUI variable synchronization."""

    setting = CameraSetting('gain', None, None, 5)
    setting.tk_var = FakeTkVar()

    setting.value = 8

    self.assertEqual(setting.tk_var.calls, [8])

  def test_reload_is_noop_on_base_setting(self) -> None:
    """Checks base reload default implementation."""

    self.assertIsNone(CameraSetting('gain', None, None, 5).reload())


class TestCameraBoolSetting(TestCase):
  """Unit tests for boolean Camera settings."""

  def test_bool_values_are_accepted(self) -> None:
    """Checks setting a valid bool value."""

    calls = list()
    setting = CameraBoolSetting('enabled', setter=calls.append, default=True)

    setting.value = False

    self.assertIs(setting.value, False)
    self.assertEqual(calls, [False])

  def test_non_bool_values_are_rejected(self) -> None:
    """Checks that non-bool values cannot be set."""

    setting = CameraBoolSetting('enabled')

    with self.assertRaises(TypeError):
      setting.value = 'yes'

    self.assertIs(setting.value, True)


class TestCameraChoiceSetting(TestCase):
  """Unit tests for choice Camera settings."""

  def test_empty_choices_are_rejected(self) -> None:
    """Checks constructor validation."""

    with self.assertRaises(ValueError):
      CameraChoiceSetting('mode', tuple())

  def test_default_is_first_choice_when_not_given(self) -> None:
    """Checks implicit default value."""

    setting = CameraChoiceSetting('mode', ('a', 'b'))

    self.assertEqual(setting.default, 'a')
    self.assertEqual(setting.value, 'a')
    self.assertEqual(setting.choices, ('a', 'b'))

  def test_invalid_default_falls_back_to_first_choice(self) -> None:
    """Checks invalid default recovery."""

    setting = CameraChoiceSetting('mode', ('a', 'b'), default='missing')

    self.assertEqual(setting.default, 'a')

  def test_reload_rejects_empty_choices(self) -> None:
    """Checks reload validation."""

    setting = CameraChoiceSetting('mode', ('a', 'b'))

    with self.assertRaises(ValueError):
      setting.reload(tuple())

  def test_reload_updates_choices_and_preserves_valid_default(self) -> None:
    """Checks default preservation when the previous default remains valid."""

    setting = CameraChoiceSetting('mode', ('a', 'b', 'c'), default='c')

    setting.reload(('b', 'c'))

    self.assertEqual(setting.choices, ('b', 'c'))
    self.assertEqual(setting.default, 'c')

  def test_reload_replaces_invalid_current_value_by_default(self) -> None:
    """Checks value replacement when current value is not a new choice."""

    setting = CameraChoiceSetting('mode', ('a', 'b'), default='a')
    setting.value = 'a'

    setting.reload(('b', 'c'), default='c')

    self.assertEqual(setting.default, 'c')
    self.assertEqual(setting.value, 'c')

  def test_reload_value_before_set_only_updates_default(self) -> None:
    """Checks delayed value application before set_all call order."""

    setting = CameraChoiceSetting('mode', ('a', 'b'))

    setting.reload(('a', 'b'), value='b')

    self.assertEqual(setting.default, 'b')
    self.assertEqual(setting.value, 'a')

  def test_reload_user_set_conflict_is_rejected(self) -> None:
    """Checks that user kwargs are not silently overridden."""

    setting = CameraChoiceSetting('mode', ('a', 'b'))
    setting.value = 'a'
    setting.user_set = True

    with self.assertRaises(ValueError):
      setting.reload(('a', 'b'), value='b')

  def test_reload_updates_radio_buttons(self) -> None:
    """Checks GUI radio button synchronization."""

    setting = CameraChoiceSetting('mode', ('a', 'b', 'c'))
    setting.tk_obj = [FakeButton(), FakeButton(), FakeButton()]

    setting.reload(('x', 'y'))

    self.assertEqual(setting.tk_obj[0].configs[-1],
                     {'value': 'x', 'text': 'x', 'state': 'normal'})
    self.assertEqual(setting.tk_obj[1].configs[-1],
                     {'value': 'y', 'text': 'y', 'state': 'normal'})
    self.assertEqual(setting.tk_obj[2].configs[-1],
                     {'state': 'disabled', 'value': '', 'text': ''})


class TestCameraScaleSetting(TestCase):
  """Unit tests for scale Camera settings."""

  def test_equal_bounds_are_rejected(self) -> None:
    """Checks constructor bound validation."""

    with self.assertRaises(ValueError):
      CameraScaleSetting('gain', 1, 1)

  def test_zero_and_negative_steps_are_rejected(self) -> None:
    """Checks constructor step validation."""

    with self.assertRaises(ValueError):
      CameraScaleSetting('gain', 0, 10, step=0)
    with self.assertRaises(ValueError):
      CameraScaleSetting('gain', 0, 10, step=-1)

  def test_reversed_bounds_are_swapped(self) -> None:
    """Checks bounds normalization."""

    setting = CameraScaleSetting('gain', 10, 0, default=2)

    self.assertEqual(setting.lowest, 0)
    self.assertEqual(setting.highest, 10)
    self.assertEqual(setting.value, 2)

  def test_int_setting_defaults_to_center_and_step_one(self) -> None:
    """Checks integer default and step handling."""

    setting = CameraScaleSetting('gain', 0, 10)

    self.assertIs(setting.type, int)
    self.assertEqual(setting.default, 5)
    self.assertEqual(setting.step, 1)

  def test_float_setting_defaults_to_center_and_small_step(self) -> None:
    """Checks float default and step handling."""

    setting = CameraScaleSetting('gain', 0., 1.)

    self.assertIs(setting.type, float)
    self.assertEqual(setting.default, 0.5)
    self.assertEqual(setting.step, 0.001)

  def test_invalid_default_falls_back_to_center(self) -> None:
    """Checks out-of-range default handling."""

    setting = CameraScaleSetting('gain', 0, 10, default=99)

    self.assertEqual(setting.default, 5)

  def test_float_step_on_int_setting_is_normalized(self) -> None:
    """Checks int setting step conversion."""

    setting = CameraScaleSetting('gain', 0, 10, step=0.5)

    self.assertEqual(setting.step, 1)

  def test_large_step_is_normalized(self) -> None:
    """Checks oversized step handling."""

    setting = CameraScaleSetting('gain', 0, 10, step=20)

    self.assertEqual(setting.step, 1)

  def test_int_highest_is_adjusted_to_step(self) -> None:
    """Checks highest value compatibility with integer step."""

    setting = CameraScaleSetting('gain', 0, 10, step=4)

    self.assertEqual(setting.highest, 8)

  def test_value_is_clamped_cast_and_forwarded_to_setter(self) -> None:
    """Checks scale setter behavior."""

    calls = list()
    setting = CameraScaleSetting('gain', 0, 10, setter=calls.append)
    setting.tk_var = FakeTkVar()

    setting.value = 12

    self.assertEqual(setting.value, 10)
    self.assertEqual(calls, [10])
    self.assertEqual(setting.tk_var.calls, [10])

  def test_getter_value_is_clamped_and_cast(self) -> None:
    """Checks scale getter behavior."""

    setting = CameraScaleSetting('gain', 0, 10, getter=lambda: 99)

    self.assertEqual(setting.value, 10)

  def test_reload_rejects_invalid_bounds_and_steps(self) -> None:
    """Checks reload validation."""

    setting = CameraScaleSetting('gain', 0, 10)

    with self.assertRaises(ValueError):
      setting.reload(1, 1)
    with self.assertRaises(ValueError):
      setting.reload(0, 10, step=0)
    with self.assertRaises(ValueError):
      setting.reload(0, 10, step=-1)

  def test_reload_recomputes_numeric_type(self) -> None:
    """Checks reloading an int setting into a float setting."""

    setting = CameraScaleSetting('gain', 0, 10)
    setting.value = 5

    setting.reload(0., 1., value=0.5, default=0.5)

    self.assertIs(setting.type, float)
    self.assertEqual(setting.default, 0.5)
    self.assertEqual(setting.value, 0.5)

  def test_reload_updates_scale_widget(self) -> None:
    """Checks GUI scale synchronization."""

    setting = CameraScaleSetting('gain', 0, 10)
    setting.tk_obj = FakeScale()

    setting.reload(1, 5, step=2)

    self.assertEqual(setting.tk_obj.configs[-1],
                     {'to': 5, 'from_': 1, 'resolution': 2})

  def test_reload_user_set_conflict_is_rejected(self) -> None:
    """Checks that user kwargs are not silently overridden."""

    setting = CameraScaleSetting('gain', 0, 10)
    setting.value = 3
    setting.user_set = True

    with self.assertRaises(ValueError):
      setting.reload(0, 10, value=4)
