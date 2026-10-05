# coding: utf-8

import unittest

from crappy.camera.meta_camera.camera_setting import (
  CameraBoolSetting, CameraChoiceSetting, CameraScaleSetting)
from crappy.tool.camera_config.base._setting_manager import SettingManager


class TestSettingManager(unittest.TestCase):
  """Headless checks for applying requested camera-configuration values."""

  def test_effective_value_and_dependent_reloads(self) -> None:
    dependent = CameraScaleSetting('dependent', -10.0, 10.0,
                                   default=0.0, step=0.1)
    choice = CameraChoiceSetting('choice', ('old', 'other'))

    def set_primary(_: int) -> None:
      dependent.reload(-1.0, 1.0, step=0.5)
      choice.reload(('new', 'other'), value='new')

    primary = CameraScaleSetting('primary', 0, 10, setter=set_primary,
                                 default=0)
    manager = SettingManager({'primary': primary, 'dependent': dependent,
                              'choice': choice})

    result = manager.apply(primary, 4)

    self.assertEqual(result.effective, 4)
    self.assertEqual(result.changed, (primary, dependent, choice))
    self.assertEqual(dependent.value, 0.0)
    self.assertEqual(dependent.step, 0.5)
    self.assertEqual(choice.value, 'new')

  def test_readback_returns_effective_clamped_value(self) -> None:
    state = {'value': 0}
    setting = CameraScaleSetting(
      'limited', 0, 10, getter=lambda: state['value'],
      setter=lambda value: state.update(value=min(value, 3)), default=0)
    manager = SettingManager({'limited': setting})

    result = manager.apply(setting, 8)

    self.assertEqual(result.effective, 3)
    self.assertEqual(result.changed, (setting,))

  def test_unchanged_request_does_not_call_setter(self) -> None:
    calls = []
    setting = CameraBoolSetting('enabled', setter=calls.append, default=True)
    manager = SettingManager({'enabled': setting})

    result = manager.apply(setting, True)

    self.assertTrue(result.effective)
    self.assertEqual(result.changed, ())
    self.assertEqual(calls, [])

  def test_unrelated_pending_edit_is_not_reported_for_sync(self) -> None:
    """Applying one setting leaves an unrelated editor request untouched."""

    enabled = CameraBoolSetting('enabled', default=False)
    gain = CameraScaleSetting('gain', 0, 10, default=0)
    manager = SettingManager({'enabled': enabled, 'gain': gain})
    pending = {enabled: False, gain: 8}
    gain_revision = gain.revision

    result = manager.apply(enabled, True)
    for setting in result.changed:
      pending[setting] = setting.value

    self.assertEqual(result.changed, (enabled,))
    self.assertEqual(gain.revision, gain_revision)
    self.assertEqual(pending, {enabled: True, gain: 8})

  def test_local_settings_precede_camera_settings(self) -> None:
    first = CameraBoolSetting('first')
    second = CameraBoolSetting('second')
    local = CameraScaleSetting('local', 0, 10, default=2)
    camera_settings = {'first': first, 'second': second}
    manager = SettingManager(camera_settings, (local,))

    self.assertEqual(manager.settings, (local, first, second))
    manager.register_local(local)
    manager.register_local(first)
    self.assertEqual(manager.local_settings, (local,))

    third = CameraBoolSetting('third')
    camera_settings['third'] = third
    self.assertEqual(manager.settings, (local, first, second, third))
    manager.register_local(third)
    self.assertEqual(manager.local_settings, (local,))

  def test_apply_rejects_unregistered_setting(self) -> None:
    manager = SettingManager({})

    with self.assertRaises(ValueError):
      manager.apply(CameraBoolSetting('unknown'), False)
