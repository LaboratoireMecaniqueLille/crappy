# coding: utf-8

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraParams)


class TestAutoApply(ConfigurationWindowTestBase):
  """Class for testing the auto-apply feature of the configuration window.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used for passing a different Camera for generating images."""

    self._camera = FakeTestCameraParams()
    self._camera.open()
    super().__init__(*args, camera=self._camera, **kwargs)

  def test_auto_apply(self) -> None:
    """Tests whether the parameters are updated as expected with and without
    the auto-apply feature."""

    # Necessary here as the callbacks are normally bound to mouse release
    self.setting_control('scale_int_setting').widget.configure(
        command=self._config._auto_apply_settings)
    self.setting_control('scale_float_setting').widget.configure(
        command=self._config._auto_apply_settings)
    self._config.update()

    # Checking that the default values were correctly passed to tkinter objects
    self.assertTrue(
        self.setting_control('bool_setting').variable.get())
    self.assertEqual(
        self.setting_control('scale_int_setting').variable.get(), 0)
    self.assertEqual(
        self.setting_control('scale_float_setting').variable.get(), 0.)
    self.assertEqual(
        self.setting_control('choice_setting').variable.get(), 'choice_1')

    # The camera settings should have the same values
    self.assertTrue(self._camera.settings['bool_setting'].value)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 0)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 0.0)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_1')

    # By default, the auto apply button should be disabled and the apply
    # settings button should be enabled
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._update_button.cget('state'), 'normal')

    # Checking the auto apply button
    self._config._auto_apply_button.invoke()

    # Now the auto apply variable should be enabled, and the apply button
    # disabled
    self.assertTrue(self._config._auto_apply.get())
    self.assertEqual(self._config._update_button.cget('state'), 'disabled')

    # Changing the values of all the parameters in the interface should be
    # automatically reflected on both the tkinter object and the camera setting
    self.setting_control('bool_setting').widget.invoke()
    self.assertFalse(self._camera.settings['bool_setting'].value)
    self.assertFalse(self.setting_control('bool_setting').variable.get())

    # Int scale setting
    self.setting_control('scale_int_setting').widget.set(4)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 0)
    self.assertEqual(
        self.setting_control('scale_int_setting').variable.get(), 4)
    # For sliders, an update is necessary for the settings to be applied
    self._config.update()
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 4)

    # Float scale setting
    self.setting_control('scale_float_setting').widget.set(4.1)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 0.0)
    self.assertEqual(
        self.setting_control('scale_float_setting').variable.get(), 4.1)
    # For sliders, an update is necessary for the settings to be applied
    self._config.update()
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)

    # Choice setting
    self.setting_control('choice_setting').widget[2].invoke()
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_3')
    self.assertEqual(
        self.setting_control('choice_setting').variable.get(), 'choice_3')

    # The values displayed in the interface should also have been updated
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.get(), 4)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.get(), 4.1)

    # Unchecking the auto apply button
    self._config._auto_apply_button.invoke()

    # The interface should be back to default
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._update_button.cget('state'), 'normal')

    # Updating the values of the parameters in the interface again
    self.setting_control('bool_setting').widget.invoke()
    self.setting_control('scale_int_setting').widget.set(6)
    self.setting_control('scale_float_setting').widget.set(3.5)
    self.setting_control('choice_setting').widget[1].invoke()

    # The values displayed in the interface should have been updated
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.get(), 6)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.get(), 3.5)

    # But not the values of the settings
    self.assertFalse(self._camera.settings['bool_setting'].value)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 4)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_3')
