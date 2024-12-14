# coding: utf-8

from time import sleep

import numpy.dtypes as np_dt

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestLoopImg(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def test_loop_img(self) -> None:
    """"""

    # Monitoring variables should be initialized to their default values
    self.assertEqual(self._config._fps_var.get(), 0.)
    self.assertEqual(self._config._nb_bits.get(), 0)
    self.assertEqual(self._config._max_pixel.get(), 0)
    self.assertEqual(self._config._min_pixel.get(), 0)
    self.assertEqual(self._config._reticle_val.get(), 0)
    self.assertEqual(self._config._x_pos.get(), 0)
    self.assertEqual(self._config._y_pos.get(), 0)
    self.assertFalse(self._config._auto_range.get())
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._zoom_level.get(), 100.0)

    # Displayed texts should be initialized to their default values
    self.assertEqual(self._config._fps_txt.get(),
                     f'fps = 0.00\n(might be lower in this GUI than actual)')
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 0')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 0, max: 0')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 0')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # Loop-related parameters should be initialized to their default values
    self.assertEqual(self._config._n_loops, 0)
    self.assertFalse(self._config._got_first_img)

    # Image-type related parameters should be None
    self.assertIsNone(self._config.dtype)
    self.assertIsNone(self._config.shape)

    # Various image containers should be initialized empty
    self.assertIsNone(self._config._img)
    self.assertIsNone(self._config._original_img)
    self.assertIsNone(self._config._pil_img)
    self.assertIsNone(self._config._hist)
    self.assertIsNone(self._config._pil_hist)

    # Sleeping to ensure the loop counter will be reset (it is every 0.5s)
    sleep(0.5)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # These monitoring variables should change because of the acquired image
    self.assertGreater(self._config._fps_var.get(), 0.)
    self.assertEqual(self._config._nb_bits.get(), 8)
    self.assertEqual(self._config._min_pixel.get(), 1)
    self.assertEqual(self._config._max_pixel.get(), 255)
    self.assertEqual(self._config._reticle_val.get(), 1)
    # All other monitoring variables should be unchanged
    self.assertEqual(self._config._x_pos.get(), 0)
    self.assertEqual(self._config._y_pos.get(), 0)
    self.assertFalse(self._config._auto_range.get())
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._zoom_level.get(), 100.0)

    # These displayed texts should change because of the acquired image
    self.assertNotEqual(self._config._fps_txt.get(),
                        f'fps = 0.00\n(might be lower in this GUI than '
                        f'actual)')
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 1, max: 255')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 1')
    # All other displayed texts should be unchanged
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # The loop counter should be reset
    self.assertEqual(self._config._n_loops, 0)
    # The first image flag should be raised
    self.assertTrue(self._config._got_first_img)

    # Image-type related parameters should no longer be None
    self.assertIsInstance(self._config.dtype, np_dt.UInt8DType)
    self.assertEqual(self._config.shape, (320, 240))

    # These image containers should not be empty
    self.assertIsNotNone(self._config._img)
    self.assertIsNotNone(self._config._original_img)
    self.assertIsNotNone(self._config._pil_img)
    # No histogram should be displayed, this takes at least 2 successful loops
    self.assertIsNone(self._config._hist)
    self.assertIsNone(self._config._pil_hist)

    # Sleeping to give some time for the histogram process to work
    sleep(2)
    # Calling a second loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # Monitoring variables should be unchanged compared to previous loop
    self.assertGreater(self._config._fps_var.get(), 0.)
    self.assertEqual(self._config._nb_bits.get(), 8)
    self.assertEqual(self._config._min_pixel.get(), 1)
    self.assertEqual(self._config._max_pixel.get(), 255)
    self.assertEqual(self._config._reticle_val.get(), 1)
    self.assertEqual(self._config._x_pos.get(), 0)
    self.assertEqual(self._config._y_pos.get(), 0)
    self.assertFalse(self._config._auto_range.get())
    self.assertFalse(self._config._auto_apply.get())
    self.assertEqual(self._config._zoom_level.get(), 100.0)

    # Displayed texts should be unchanged compared to previous loop
    self.assertNotEqual(self._config._fps_txt.get(),
                        f'fps = 0.00\n(might be lower in this GUI than '
                        f'actual)')
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 1, max: 255')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 1')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # The loop counter should have been reset
    self.assertEqual(self._config._n_loops, 0)
    self.assertTrue(self._config._got_first_img)

    # Image-type parameters should be unchanged compared to previous loop
    self.assertIsInstance(self._config.dtype, np_dt.UInt8DType)
    self.assertEqual(self._config.shape, (320, 240))

    # The histogram should now have been loaded
    self.assertIsNotNone(self._config._img)
    self.assertIsNotNone(self._config._original_img)
    self.assertIsNotNone(self._config._pil_img)
    self.assertIsNotNone(self._config._hist)
    self.assertIsNotNone(self._config._pil_hist)
