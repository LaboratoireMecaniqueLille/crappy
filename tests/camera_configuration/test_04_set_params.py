# coding: utf-8

from time import sleep

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraParams)


class TestSetParams(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    self._camera = FakeTestCameraParams()
    self._camera.open()
    super().__init__(*args, camera=self._camera, **kwargs)

  def test_set_params(self) -> None:
    """"""

    # All the tkinter objects of the parameters should have been set
    self.assertIsNotNone(self._camera.settings['bool_setting'].tk_var)
    self.assertIsNotNone(self._camera.settings['bool_setting'].tk_obj)
    self.assertIsNotNone(self._camera.settings['scale_int_setting'].tk_var)
    self.assertIsNotNone(self._camera.settings['scale_int_setting'].tk_obj)
    self.assertIsNotNone(self._camera.settings['scale_float_setting'].tk_var)
    self.assertIsNotNone(self._camera.settings['scale_float_setting'].tk_obj)
    self.assertIsNotNone(self._camera.settings['choice_setting'].tk_var)
    self.assertEqual(len(self._camera.settings['choice_setting'].tk_obj), 3)

    # Checking if the parameters of the tkinter objects match those given in
    # the Camera object
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('from'),
        self._camera._scale_int_bounds[0])
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('to'),
        self._camera._scale_int_bounds[1])
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('resolution'),
        self._camera._scale_int_bounds[2])
    self.assertEqual(
      self._camera.settings['scale_float_setting'].tk_obj.cget('from'),
      self._camera._scale_float_bounds[0])
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.cget('to'),
        self._camera._scale_float_bounds[1])
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.cget('resolution'),
        self._camera._scale_float_bounds[2])
    for i in range(3):
      self.assertEqual(
          self._camera.settings['choice_setting'].tk_obj[i].cget('value'),
          self._camera._choices[i])

    # Checking that the default values were correctly passed to tkinter objects
    self.assertIsInstance(
        self._camera.settings['bool_setting'].tk_var.get(), bool)
    self.assertTrue(
        self._camera.settings['bool_setting'].tk_var.get())
    self.assertIsInstance(
        self._camera.settings['scale_int_setting'].tk_var.get(), int)
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_var.get(), 0)
    self.assertIsInstance(
        self._camera.settings['scale_float_setting'].tk_var.get(), float)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_var.get(), 0.)
    self.assertIsInstance(
        self._camera.settings['choice_setting'].tk_var.get(), str)
    self.assertEqual(
        self._camera.settings['choice_setting'].tk_var.get(), 'choice_1')

    # At that point, the getter should have been called but not the setter
    self.assertTrue(self._camera._bool_getter_called)
    self.assertFalse(self._camera._bool_setter_called)
    self.assertTrue(self._camera._scale_int_getter_called)
    self.assertFalse(self._camera._scale_int_setter_called)
    self.assertTrue(self._camera._scale_float_getter_called)
    self.assertFalse(self._camera._scale_float_setter_called)
    self.assertTrue(self._camera._choice_getter_called)
    self.assertFalse(self._camera._choice_setter_called)

    # Changing the values of all the parameters in the interface
    self._camera.settings['bool_setting'].tk_obj.invoke()
    self._camera.settings['scale_int_setting'].tk_obj.set(4)
    self._camera.settings['scale_float_setting'].tk_obj.set(4.1)
    self._camera.settings['choice_setting'].tk_obj[2].invoke()

    # The values displayed in the interface should have been updated
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.get(), 4)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.get(), 4.1)

    # At that point the setters still shouldn't have been called
    self.assertFalse(self._camera._bool_setter_called)
    self.assertFalse(self._camera._scale_int_setter_called)
    self.assertFalse(self._camera._scale_float_setter_called)
    self.assertFalse(self._camera._choice_setter_called)

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Looping once
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

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
    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

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

    # Reloading the settings that support it
    self._camera.settings['scale_int_setting'].reload(-50, 50, 42)
    self._camera.settings['scale_float_setting'].reload(-5.0, 5.0, 4.2)
    self._camera.settings['choice_setting'].reload(('choice_4', 'choice_5'))

    # The bounds of the tkinter objects should have been updated
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('from'), -50)
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('to'), 50)
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.cget('resolution'),
        1)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.cget('from'), -5.0)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.cget('to'), 5.0)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.cget('resolution'),
        0.01)
    for i, val in enumerate(('choice_4', 'choice_5')):
      self.assertEqual(
          self._camera.settings['choice_setting'].tk_obj[i].cget('value'), val)
    self.assertEqual(
      self._camera.settings['choice_setting'].tk_obj[2].cget('state'),
      'disabled')

    # The values displayed in the interface should have been updated as well
    self.assertEqual(
        self._camera.settings['scale_int_setting'].tk_obj.get(), 42)
    self.assertEqual(
        self._camera.settings['scale_float_setting'].tk_obj.get(), 4.2)

    # And the values of the settings as well
    self.assertEqual(self._camera.settings['scale_int_setting'].value, 42)
    self.assertEqual(self._camera.settings['scale_float_setting'].value, 4.2)
    self.assertEqual(self._camera.settings['choice_setting'].value, 'choice_4')
