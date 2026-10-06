# coding: utf-8

"""Qt settings, deferred Apply, reloads, and scale-value formatting."""

from crappy.camera.meta_camera.camera_setting import (CameraBoolSetting,
                                                      CameraChoiceSetting,
                                                      CameraScaleSetting)
from crappy.tool.camera_config import SpotsBoxes
from crappy.tool.camera_config.pyqt import PyQtDICVEConfig
from ._fixtures import PyQtConfigTestCase


class TestSettings(PyQtConfigTestCase):
  def test_pending_edits_are_deferred_until_apply(self) -> None:
    """All four setting types stay unchanged until the Apply button is used."""

    enabled = CameraBoolSetting('enabled', default=False)
    gain = CameraScaleSetting('gain', 0, 10, default=2, step=2)
    exposure = CameraScaleSetting('exposure', 0.0, 10.0,
                                 default=2.5, step=0.25)
    mode = CameraChoiceSetting('mode', ('a', 'b'), default='a')
    self.camera.settings.update(enabled=enabled, gain=gain,
                                exposure=exposure, mode=mode)
    config = self.make_config()

    config._setting_controls[enabled].widget.click()
    config._setting_controls[gain].slider.setValue(3)
    config._setting_controls[exposure].slider.setValue(14)
    config._setting_controls[mode].buttons.buttons()[1].click()
    self.assertEqual((enabled.value, gain.value, exposure.value, mode.value),
                     (False, 2, 2.5, 'a'))
    config._apply_button.click()
    self.assertEqual((enabled.value, gain.value, exposure.value, mode.value),
                     (True, 6, 3.5, 'b'))

  def test_choice_reload_rebuilds_radio_buttons(self) -> None:
    """A reload replaces the available choices and the selected value."""

    mode = CameraChoiceSetting('mode', ('a', 'b'), default='a')
    self.camera.settings['mode'] = mode
    config = self.make_config()

    mode.reload(('c', 'd'), value='d')
    config._sync_setting_controls()
    self.assertEqual([button.text() for button in
                      config._setting_controls[mode].buttons.buttons()],
                     ['c', 'd'])
    self.assertEqual(config._setting_controls[mode].buttons.checkedButton()
                     .property('choice'), 'd')

  def test_auto_apply_handles_widgets_and_toggle(self) -> None:
    """Auto apply handles checkbox, slider release, and radio-button signals."""

    enabled = CameraBoolSetting('enabled', default=False)
    gain = CameraScaleSetting('gain', 0, 10, default=2, step=2)
    mode = CameraChoiceSetting('mode', ('a', 'b'), default='a')
    self.camera.settings.update(enabled=enabled, gain=gain, mode=mode)
    config = self.make_config()
    config._auto_apply_button.click()
    self.assertFalse(config._apply_button.isEnabled())
    config._setting_controls[enabled].widget.click()
    config._setting_controls[gain].slider.setValue(4)
    config._setting_controls[gain].slider.sliderReleased.emit()
    config._setting_controls[mode].buttons.buttons()[1].click()
    self.assertEqual((enabled.value, gain.value, mode.value), (True, 8, 'b'))

    config._auto_apply_button.click()
    self.assertTrue(config._apply_button.isEnabled())
    config._setting_controls[gain].slider.setValue(3)
    config._setting_controls[gain].slider.sliderReleased.emit()
    self.assertEqual(gain.value, 8)

  def test_unrelated_pending_edit_survives_model_sync(self) -> None:
    """Synchronizing one setting cannot discard a different pending edit."""

    enabled = CameraBoolSetting('enabled', default=False)
    gain = CameraScaleSetting('gain', 0, 10, default=2, step=2)
    self.camera.settings.update(enabled=enabled, gain=gain)
    config = self.make_config()
    config._setting_controls[gain].slider.setValue(3)
    enabled.value = True
    config._sync_setting_controls()

    self.assertTrue(config._setting_controls[enabled].widget.isChecked())
    self.assertEqual(config._setting_controls[gain].slider.value(), 3)
    self.assertEqual(gain.value, 2)

  def test_control_shows_effective_setter_readback(self) -> None:
    """A Camera that clamps a request leaves its effective value in the UI."""

    state = {'value': 0}
    gain = CameraScaleSetting('gain', 0, 10,
                              getter=lambda: state['value'],
                              setter=lambda value: state.update(value=min(3, value)),
                              default=0)
    self.camera.settings['gain'] = gain
    config = self.make_config()
    config._setting_controls[gain].slider.setValue(8)
    config._apply_button.click()

    self.assertEqual(gain.value, 3)
    self.assertEqual(config._setting_controls[gain].slider.value(), 3)
    self.assertEqual(config._setting_controls[gain].value_label.text(),
                     'gain : 3')

  def test_reload_can_change_scale_type(self) -> None:
    """Qt sliders follow changes to type, bounds, step, and effective value."""

    gain = CameraScaleSetting('gain', 0, 10, default=2)
    self.camera.settings['gain'] = gain
    config = self.make_config()
    gain.reload(0.0, 1.0, value=0.375, step=0.125)
    config._sync_setting_controls()
    control = config._setting_controls[gain]

    self.assertEqual(control.slider.maximum(), 8)
    self.assertEqual(control.slider.value(), 3)
    self.assertEqual(config._requested_value(gain, control), 0.375)
    self.assertEqual(control.value_label.text(), 'gain : 0.375')

  def test_local_settings_precede_camera_setting_setters(self) -> None:
    """The DICVE patch-size control uses the shared Apply path first."""

    observed = []
    self.camera.add_bool_setting(
        'capture_patch_size', setter=lambda _: observed.append(
            config._patch_size.value))
    config = self.make_config(PyQtDICVEConfig, SpotsBoxes())
    local = config._patch_size
    capture = self.camera.settings['capture_patch_size']
    config._setting_controls[local].slider.setValue(18)
    config._setting_controls[capture].widget.click()
    config._apply_button.click()

    self.assertIs(config._setting_manager.settings[0], local)
    self.assertEqual(local.value, 20)
    self.assertEqual(observed, [20])

  def test_setting_reload_refreshes_a_later_pending_edit(self) -> None:
    """A dependent reload wins over a stale slider edit during Apply."""

    gain = CameraScaleSetting('gain', 0, 10, default=2, step=2)

    def enabled_changed(value: bool) -> None:
      if value:
        gain.reload(0, 10, value=8, step=2)

    enabled = CameraBoolSetting('enabled', setter=enabled_changed,
                                default=False)
    self.camera.settings.update(enabled=enabled, gain=gain)
    config = self.make_config()
    config._setting_controls[enabled].widget.click()
    config._setting_controls[gain].slider.setValue(3)
    config._apply_button.click()

    self.assertTrue(enabled.value)
    self.assertEqual(gain.value, 8)
    self.assertEqual(config._setting_controls[gain].slider.value(), 4)

  def test_scale_labels_limit_float_digits(self) -> None:
    """Initial, pending, and reloaded values use the scale's precision."""

    gain = CameraScaleSetting('gain', 0.0, 1.0, default=0.3, step=0.1)
    self.camera.settings['gain'] = gain
    config = self.make_config()
    control = config._setting_controls[gain]
    self.assertEqual(control.value_label.text(), 'gain : 0.3')
    control.slider.setValue(7)
    self.assertEqual(control.value_label.text(), 'gain : 0.7')
    self.assertEqual(config._requested_value(gain, control), 7 * gain.step)
    self.assertEqual(gain.value, 0.3)

    gain.reload(0.125, 1.125, step=0.25, value=0.875)
    config._sync_setting_controls()
    self.assertEqual(control.value_label.text(), 'gain : 0.875')
    control.slider.setValue(1)
    self.assertEqual(control.value_label.text(), 'gain : 0.375')

  def test_scale_labels_preserve_small_values_and_integers(self) -> None:
    """Compact labels retain fractional steps, small scales, and integers."""

    from crappy.tool.camera_config.pyqt import PyQtCameraConfig

    cases = ((0.0, 1.0, 0.25, 0.75, '0.75'),
             (0.0, 1e-9, 1e-10, 3e-10, '3e-10'),
             (0.0, 1.0, None, 0.30000000000000004, '0.3'),
             (-1.0, 1.0, 0.1, -0.30000000000000004, '-0.3'),
             (0, 123456789, 1, 123456789, '123456789'))
    for lowest, highest, step, value, label in cases:
      with self.subTest(step=step, value=value):
        setting = CameraScaleSetting('scale', lowest, highest, step=step)
        self.assertEqual(PyQtCameraConfig._scale_label(setting, value),
                         f'scale : {label}')
