"""Public validation and protected state for this feature."""

from unittest import TestCase
import numpy as np
from crappy.camera.meta_camera.camera_setting import CameraChoiceSetting, CameraScaleSetting


class TestSettingValidation(TestCase):
  def test_camera_settings_reject_bad_types_before_side_effects(self):
    calls = []
    setting = CameraScaleSetting('x', 0, 10, setter=calls.append)
    revision = setting.revision
    for value, error in ((np.nan, ValueError), (np.inf, ValueError),
                          ('1', TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          setting.value = value
    self.assertEqual(calls, [])
    self.assertEqual(setting.revision, revision)
    with self.assertRaises((TypeError, ValueError)):
      setting.reload(0, np.nan)
    self.assertEqual((setting.lowest, setting.highest), (0, 10))

  def test_choice_settings_validate_assignments_and_reload_atomically(self):
    setting = CameraChoiceSetting('mode', ('a', 'b'))
    for value, error in (('missing', ValueError), (1, TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          setting.value = value
    with self.assertRaises((TypeError, ValueError)):
      setting.reload(('a', 1))
    self.assertEqual(setting.choices, ('a', 'b'))
    self.assertEqual(setting.value, 'a')
