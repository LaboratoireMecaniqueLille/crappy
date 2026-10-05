# coding: utf-8

"""Tk controls, deferred Apply, Auto apply, and dependent setting reloads."""

import locale

from ._fixtures import TkinterConfigTestCase, FakeTestCameraParams
from crappy.camera.meta_camera.camera_setting import CameraScaleSetting


class TestNumericLocale(TkinterConfigTestCase):
  """Floating-point controls must work after another GUI changes the locale."""

  def make_camera(self) -> FakeTestCameraParams:
    """Create a Camera with integer and floating-point settings."""

    camera = FakeTestCameraParams()
    camera.open()
    return camera

  def customSetUp(self) -> None:
    """Start the window with a comma-based numeric locale already selected."""

    previous = locale.setlocale(locale.LC_NUMERIC)
    self.addCleanup(locale.setlocale, locale.LC_NUMERIC, previous)
    for numeric_locale in ('fr_FR.UTF-8', 'French_France.1252'):
      try:
        locale.setlocale(locale.LC_NUMERIC, numeric_locale)
      except locale.Error:
        continue
      break
    else:
      self.skipTest('a French numeric locale is required')

    self.assertEqual(locale.localeconv()['decimal_point'], ',')
    self._text_locale = locale.setlocale(locale.LC_CTYPE)
    super().customSetUp()

  def test_float_scale_after_comma_numeric_locale(self) -> None:
    """Float values remain readable, can be applied, and survive a reload."""

    control = self.setting_control('scale_float_setting')
    control.widget.set(4.1)
    self.assertEqual(control.widget.get(), 4.1)
    self.assertEqual(control.variable.get(), 4.1)
    self._config._update_button.invoke()
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)

    self._camera.settings['scale_float_setting'].reload(-5.0, 5.0, 4.2)
    self._config._sync_setting_controls()
    self.assertEqual(control.widget.get(), 4.2)
    self.assertEqual(control.variable.get(), 4.2)
    self.assertEqual(locale.setlocale(locale.LC_NUMERIC), 'C')
    self.assertEqual(locale.setlocale(locale.LC_CTYPE), self._text_locale)


class TestSetParams(TkinterConfigTestCase):
  """Class for testing the behavior of the configuration window with camera
  parameters.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraParams:
    """Create fresh Camera settings for each test."""

    camera = FakeTestCameraParams()
    camera.open()
    return camera

  def test_controls_are_initialized_from_camera_settings(self) -> None:
    """Controls read the Camera values and metadata without applying them."""

    # All the tkinter objects of the parameters should have been set
    self.assertIsNotNone(self.setting_control('bool_setting').variable)
    self.assertIsNotNone(self.setting_control('bool_setting').widget)
    self.assertIsNotNone(self.setting_control('scale_int_setting').variable)
    self.assertIsNotNone(self.setting_control('scale_int_setting').widget)
    self.assertIsNotNone(self.setting_control('scale_float_setting').variable)
    self.assertIsNotNone(self.setting_control('scale_float_setting').widget)
    self.assertIsNotNone(self.setting_control('choice_setting').variable)
    self.assertEqual(len(self.setting_control('choice_setting').widget), 3)

    # Checking if the parameters of the tkinter objects match those given in
    # the Camera object
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('from'),
        self._camera._scale_int_bounds[0])
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('to'),
        self._camera._scale_int_bounds[1])
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('resolution'),
        self._camera._scale_int_bounds[2])
    self.assertEqual(
      self.setting_control('scale_float_setting').widget.cget('from'),
      self._camera._scale_float_bounds[0])
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.cget('to'),
        self._camera._scale_float_bounds[1])
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.cget('resolution'),
        self._camera._scale_float_bounds[2])
    for i in range(3):
      self.assertEqual(
          self.setting_control('choice_setting').widget[i].cget('value'),
          self._camera._choices[i])

    # Checking that the default values were correctly passed to tkinter objects
    self.assertIsInstance(
        self.setting_control('bool_setting').variable.get(), bool)
    self.assertTrue(
        self.setting_control('bool_setting').variable.get())
    self.assertIsInstance(
        self.setting_control('scale_int_setting').variable.get(), int)
    self.assertEqual(
        self.setting_control('scale_int_setting').variable.get(), 0)
    self.assertIsInstance(
        self.setting_control('scale_float_setting').variable.get(), float)
    self.assertEqual(
        self.setting_control('scale_float_setting').variable.get(), 0.)
    self.assertIsInstance(
        self.setting_control('choice_setting').variable.get(), str)
    self.assertEqual(
        self.setting_control('choice_setting').variable.get(), 'choice_1')

    # At that point, the getter should have been called but not the setter
    self.assertTrue(self._camera._bool_getter_called)
    self.assertFalse(self._camera._bool_setter_called)
    self.assertTrue(self._camera._scale_int_getter_called)
    self.assertFalse(self._camera._scale_int_setter_called)
    self.assertTrue(self._camera._scale_float_getter_called)
    self.assertFalse(self._camera._scale_float_setter_called)
    self.assertTrue(self._camera._choice_getter_called)
    self.assertFalse(self._camera._choice_setter_called)

  def test_edits_are_deferred_until_apply(self) -> None:
    """Apply sends all pending control values to the Camera setters."""

    # Changing the values of all the parameters in the interface
    self.setting_control('bool_setting').widget.invoke()
    self.setting_control('scale_int_setting').widget.set(4)
    self.setting_control('scale_float_setting').widget.set(4.1)
    self.setting_control('choice_setting').widget[2].invoke()

    # The values displayed in the interface should have been updated
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.get(), 4)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.get(), 4.1)

    # At that point the setters still shouldn't have been called
    self.assertFalse(self._camera._bool_setter_called)
    self.assertFalse(self._camera._scale_int_setter_called)
    self.assertFalse(self._camera._scale_float_setter_called)
    self.assertFalse(self._camera._choice_setter_called)

    self.run_config_cycle()

    # The setter should still not have been called as the Apply button wasn't
    # clicked and the auto apply mode isn't set
    self.assertFalse(self._camera._bool_setter_called)
    self.assertFalse(self._camera._scale_int_setter_called)
    self.assertFalse(self._camera._scale_float_setter_called)
    self.assertFalse(self._camera._choice_setter_called)

    # The camera settings should still have their original value
    self.assertTrue(self._camera.settings['bool_setting'].value)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 0)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 0.0)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_1')

    # Looping once again but this time with the update button clicked
    self._config._update_button.invoke()
    self.run_config_cycle()

    # Now all the setters should have been called
    self.assertTrue(self._camera._bool_setter_called)
    self.assertTrue(self._camera._scale_int_setter_called)
    self.assertTrue(self._camera._scale_float_setter_called)
    self.assertTrue(self._camera._choice_setter_called)

    # The values of the settings should have been updated
    self.assertFalse(self._camera.settings['bool_setting'].value)
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 4)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.1)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_3')

  def test_reload_updates_control_metadata_and_values(self) -> None:
    """Reloads update initialized scales and choices without rebuilding Tk."""

    # Initialize the settings before reloading their bounds and choices
    self.setting_control('scale_int_setting').widget.set(4)
    self.setting_control('scale_float_setting').widget.set(4.1)
    self.setting_control('choice_setting').widget[2].invoke()
    self._config._update_button.invoke()

    # Reloading the settings that support it
    self._camera.settings['scale_int_setting'].reload(-50, 50, 42)
    self._camera.settings['scale_float_setting'].reload(-5.0, 5.0, 4.2)
    self._camera.settings['choice_setting'].reload(('choice_4', 'choice_5'))
    self._config._sync_setting_controls()

    # The bounds of the tkinter objects should have been updated
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('from'), -50)
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('to'), 50)
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.cget('resolution'),
        1)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.cget('from'), -5.0)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.cget('to'), 5.0)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.cget('resolution'),
        0.01)
    for i, val in enumerate(('choice_4', 'choice_5')):
      self.assertEqual(
          self.setting_control('choice_setting').widget[i].cget('value'), val)
    self.assertEqual(
      self.setting_control('choice_setting').widget[2].cget('state'),
      'disabled')

    # The values displayed in the interface should have been updated as well
    self.assertEqual(
        self.setting_control('scale_int_setting').widget.get(), 42)
    self.assertEqual(
        self.setting_control('scale_float_setting').widget.get(), 4.2)

    # And the values of the settings as well
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 42)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.2)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_4')

  def test_scale_settings_without_step_get_default_resolution(self) -> None:
    """Tests the GUI defaults for scale settings with no explicit step."""

    self._camera.add_scale_setting('scale_int_without_step', -100, 100,
                                   default=0)
    self._camera.add_scale_setting('scale_float_without_step', -10.0, 10.0,
                                   default=0.0)

    int_setting = self._camera.settings['scale_int_without_step']
    float_setting = self._camera.settings['scale_float_without_step']
    self._config._add_slider_setting(int_setting)
    self._config._add_slider_setting(float_setting)

    self.assertEqual(int_setting.step, 1)
    self.assertEqual(
      self.setting_control('scale_int_without_step').widget.cget('resolution'),
      1)
    self.assertAlmostEqual(float_setting.step, 0.02)
    self.assertAlmostEqual(
      self.setting_control('scale_float_without_step').widget.cget('resolution'),
      0.02)

  def test_dependent_reload_updates_later_controls_before_apply(self) -> None:
    """A camera setter may replace pending values in dependent settings."""

    scale = self._camera.settings['scale_int_setting']
    other_scale = self._camera.settings['scale_float_setting']
    choice = self._camera.settings['choice_setting']
    original_setter = scale._setter

    def set_and_reload(value):
      original_setter(value)
      other_scale.reload(-5.0, 5.0, value=0.5, step=0.25)
      choice.reload(('new_1', 'new_2'), value='new_2')

    scale._setter = set_and_reload
    self.setting_control('scale_float_setting').widget.set(4.1)
    self.setting_control('choice_setting').widget[2].invoke()
    self.setting_control('scale_int_setting').widget.set(4)

    self._config._update_button.invoke()

    self.assertEqual(other_scale.value, 0.5)
    self.assertEqual(self.setting_control('scale_float_setting').variable.get(),
                     0.5)
    self.assertEqual(
      self.setting_control('scale_float_setting').widget.cget('resolution'),
      0.25)
    self.assertEqual(choice.value, 'new_2')
    self.assertEqual(self.setting_control('choice_setting').variable.get(),
                     'new_2')
    self.assertEqual(self.setting_control('choice_setting').widget[0].cget(
      'value'), 'new_1')

  def test_reload_can_expand_choices_and_change_scale_type(self) -> None:
    """The Tk controls follow metadata changes without owning the settings."""

    choice = self._camera.settings['choice_setting']
    scale = self._camera.settings['scale_int_setting']

    choice.reload(('choice_1', 'choice_2', 'choice_3', 'choice_4'),
                  value='choice_4')
    scale.reload(0.0, 1.0, value=0.5)
    self._config._sync_setting_controls()

    choice_control = self.setting_control('choice_setting')
    scale_control = self.setting_control('scale_int_setting')
    self.assertEqual(len(choice_control.widget), 4)
    self.assertEqual(choice_control.widget[3].cget('value'), 'choice_4')
    self.assertEqual(choice_control.variable.get(), 'choice_4')
    self.assertEqual(scale_control.variable.get(), 0.5)
    self.assertAlmostEqual(scale_control.widget.cget('resolution'), 0.001)

  def test_unrelated_pending_edit_survives_one_setting_apply(self) -> None:
    """Applying one control does not replace a pending edit in another."""

    choice = self._camera.settings['choice_setting']
    self.setting_control('choice_setting').widget[2].invoke()
    self.setting_control('scale_int_setting').widget.set(4)

    self._config._apply_setting(self._camera.settings['scale_int_setting'])

    self.assertEqual(choice.value, 'choice_1')
    self.assertEqual(self.setting_control('choice_setting').variable.get(),
                     'choice_3')

  def test_control_shows_effective_setter_readback(self) -> None:
    """A camera that rejects a request leaves its actual value in the UI."""

    setting = self._camera.settings['scale_int_setting']
    setting._setter = lambda value: setattr(
      setting, '_value_no_getter', min(value, 2))
    control = self.setting_control('scale_int_setting')
    control.widget.set(4)

    self._config._update_button.invoke()

    self.assertEqual(setting.value, 2)
    self.assertEqual(control.variable.get(), 2)

  def test_manually_added_local_setting_is_applied(self) -> None:
    """Existing subclasses can still add a local setting's Tk control."""

    local = CameraScaleSetting('extra local', 0, 10, default=1)
    self._config._add_slider_setting(local)
    self.assertIn(local, self._config._setting_manager.local_settings)

    self._config._setting_controls[local].widget.set(5)
    self._config._update_button.invoke()

    self.assertEqual(local.value, 5)


class TestAutoApply(TkinterConfigTestCase):
  """Class for testing the auto-apply feature of the configuration window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraParams:
    """Create fresh Camera settings for each test."""

    camera = FakeTestCameraParams()
    camera.open()
    return camera

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
    self.assertFalse(self._config._display_state.auto_apply)
    self.assertEqual(self._config._update_button.cget('state'), 'normal')

    # Checking the auto apply button
    self._config._auto_apply_button.invoke()

    # Now the auto apply variable should be enabled, and the apply button
    # disabled
    self.assertTrue(self._config._display_state.auto_apply)
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
    self.assertFalse(self._config._display_state.auto_apply)
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
