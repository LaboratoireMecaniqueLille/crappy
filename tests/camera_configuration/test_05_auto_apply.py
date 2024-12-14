# coding: utf-8

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraParams)


class TestAutoApply(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    self._camera = FakeTestCameraParams()
    self._camera.open()
    super().__init__(*args, camera=self._camera, **kwargs)

  def test_auto_apply(self) -> None:
    """"""

    # Necessary here as the callbacks are normally bound to mouse release
    self._camera.settings['scale_int_setting'].tk_obj.configure(
        command=self._config._auto_apply_settings)
    self._camera.settings['scale_float_setting'].tk_obj.configure(
        command=self._config._auto_apply_settings)
    self._config.update()

    # Checking that the default values were correctly passed to tkinter objects
    self.assertTrue(
        self._camera.settings['bool_setting'].tk_var.get())
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_var.get(), 0)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_var.get(), 0.)
    self.assertEqual(
        self._camera.settings['choice_setting'].tk_var.get(), 'choice_1')

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
    self._camera.settings['bool_setting'].tk_obj.invoke()
    self.assertFalse(self._camera.settings['bool_setting'].value)
    self.assertFalse(self._camera.settings['bool_setting'].tk_var.get())

    # Int scale setting
    self._camera.settings['scale_int_setting'].tk_obj.set(4)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 0)
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_var.get(), 4)
    # For sliders, an update is necessary for the settings to be applied
    self._config.update()
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 4)

    # Float scale setting
    self._camera.settings['scale_float_setting'].tk_obj.set(4.1)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 0.0)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_var.get(), 4.1)
    # For sliders, an update is necessary for the settings to be applied
    self._config.update()
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)

    # Choice setting
    self._camera.settings['choice_setting'].tk_obj[2].invoke()
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_3')
    self.assertEqual(
        self._camera.settings['choice_setting'].tk_var.get(), 'choice_3')

    # The values displayed in the interface should also have been updated
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.get(), 4)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.get(), 4.1)

    # Unchecking the auto apply button
    self._config._auto_apply_button.invoke()

    # The interface should be back to default
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._update_button.cget('state'), 'normal')

    # Updating the values of the parameters in the interface again
    self._camera.settings['bool_setting'].tk_obj.invoke()
    self._camera.settings['scale_int_setting'].tk_obj.set(6)
    self._camera.settings['scale_float_setting'].tk_obj.set(3.5)
    self._camera.settings['choice_setting'].tk_obj[1].invoke()

    # The values displayed in the interface should have been updated
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.get(), 6)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.get(), 3.5)

    # But not the values of the settings
    self.assertFalse(self._camera.settings['bool_setting'].value)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 4)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_3')
