# coding: utf-8

from time import sleep

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestAutoRange(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args,
                     camera=FakeTestCameraSimple(min_val=3, max_val=252),
                     **kwargs)

  def test_auto_range(self) -> None:
    """"""

    # The auto range feature should be disabled by default
    self.assertFalse(self._config._auto_range.get())

    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # Both stored images should have bounds at 3 and 255
    self.assertEqual(int(self._config._img.min()), 3)
    self.assertEqual(int(self._config._img.max()), 252)
    self.assertEqual(int(self._config._original_img.min()), 3)
    self.assertEqual(int(self._config._original_img.max()), 252)

    # Enable the auto range feature
    self._config._auto_range_button.invoke()

    # The auto range feature should be enabled
    self.assertTrue(self._config._auto_range.get())

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Call a second loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The original image should have the original limits, but the other one
    # should have been extended to use the full range
    self.assertEqual(int(self._config._img.min()), 0)
    self.assertEqual(int(self._config._img.max()), 255)
    self.assertEqual(int(self._config._original_img.min()), 3)
    self.assertEqual(int(self._config._original_img.max()), 252)

    # Disable the auto range feature
    self._config._auto_range_button.invoke()

    # The auto range feature should be disabled
    self.assertFalse(self._config._auto_range.get())

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Call a third loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The behavior should be back to default
    self.assertEqual(int(self._config._img.min()), 3)
    self.assertEqual(int(self._config._img.max()), 252)
    self.assertEqual(int(self._config._original_img.min()), 3)
    self.assertEqual(int(self._config._original_img.max()), 252)
