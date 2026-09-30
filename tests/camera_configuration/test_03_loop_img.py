# coding: utf-8

import numpy as np

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestLoopImg(ConfigurationWindowTestBase):
  """Class for testing the looping behavior of the configuration window.

  .. versionadded:: 2.0.8
  """

  start_histogram_process = True

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args,
                     camera=FakeTestCameraSimple(min_val=3, max_val=252),
                     **kwargs)

  def test_loop_img(self) -> None:
    """Tests whether the internal state variables of the configuration window
    are updated as expected when looping with an image."""

    # Monitoring variables should be initialized to their default values
    self.assertEqual(self._config._display_state.fps, 0.)
    self.assertEqual(self._config._display_state.detected_bits, 0)
    self.assertEqual(self._config._display_state.max_pixel, 0)
    self.assertEqual(self._config._display_state.min_pixel, 0)
    self.assertEqual(self._config._display_state.reticle_value, 0)
    self.assertEqual(self._config._display_state.reticle_x, 0)
    self.assertEqual(self._config._display_state.reticle_y, 0)
    self.assertFalse(self._config._display_state.auto_range)
    self.assertFalse(self._config._display_state.auto_apply)
    self.assertEqual(self._config._display_state.zoom_percent, 100.0)

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

    self.run_config_cycle(elapsed=0.5)

    # These monitoring variables should change because of the acquired image
    self.assertGreater(self._config._display_state.fps, 0.)
    self.assertEqual(self._config._display_state.detected_bits, 8)
    self.assertEqual(self._config._display_state.min_pixel, 3)
    self.assertEqual(self._config._display_state.max_pixel, 252)
    self.assertEqual(self._config._display_state.reticle_value, 3)
    # All other monitoring variables should be unchanged
    self.assertEqual(self._config._display_state.reticle_x, 0)
    self.assertEqual(self._config._display_state.reticle_y, 0)
    self.assertFalse(self._config._display_state.auto_range)
    self.assertFalse(self._config._display_state.auto_apply)
    self.assertEqual(self._config._display_state.zoom_percent, 100.0)

    # These displayed texts should change because of the acquired image
    self.assertNotEqual(self._config._fps_txt.get(),
                        f'fps = 0.00\n(might be lower in this GUI than '
                        f'actual)')
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 3, max: 252')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 3')
    # All other displayed texts should be unchanged
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # The loop counter should be reset
    self.assertEqual(self._config._n_loops, 0)
    # The first image flag should be raised
    self.assertTrue(self._config._got_first_img)

    # Image-type related parameters should no longer be None
    self.assertEqual(self._config.dtype, 'uint8')
    self.assertEqual(self._config.shape, (240, 320))

    # These image containers should not be empty
    self.assertIsNotNone(self._config._img)
    self.assertIsNotNone(self._config._original_img)
    self.assertIsNotNone(self._config._pil_img)
    # No histogram should be displayed, this takes at least 2 successful loops
    self.assertIsNone(self._config._hist)
    self.assertIsNone(self._config._pil_hist)

    self.assertTrue(self.wait_for_histogram())
    self.run_config_cycle()

    # Monitoring variables should be unchanged compared to previous loop
    self.assertGreater(self._config._display_state.fps, 0.)
    self.assertEqual(self._config._display_state.detected_bits, 8)
    self.assertEqual(self._config._display_state.min_pixel, 3)
    self.assertEqual(self._config._display_state.max_pixel, 252)
    state = self._config._display_state
    self.assertEqual(state.reticle_value, int(np.average(
      self._config._original_img[state.reticle_y, state.reticle_x])))
    self.assertFalse(self._config._display_state.auto_range)
    self.assertFalse(self._config._display_state.auto_apply)
    self.assertEqual(self._config._display_state.zoom_percent, 100.0)

    # Displayed texts should be unchanged compared to previous loop
    self.assertNotEqual(self._config._fps_txt.get(),
                        f'fps = 0.00\n(might be lower in this GUI than '
                        f'actual)')
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 3, max: 252')
    self.assertEqual(self._config._reticle_txt.get(),
                     f'X: {state.reticle_x}, Y: {state.reticle_y}, '
                     f'V: {state.reticle_value}')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # The loop counter should have been reset
    self.assertEqual(self._config._n_loops, 0)
    self.assertTrue(self._config._got_first_img)

    # Image-type parameters should be unchanged compared to previous loop
    self.assertEqual(self._config.dtype, 'uint8')
    self.assertEqual(self._config.shape, (240, 320))

    # The histogram should now have been loaded
    self.assertIsNotNone(self._config._img)
    self.assertIsNotNone(self._config._original_img)
    self.assertIsNotNone(self._config._pil_img)
    self.assertIsNotNone(self._config._hist)
    self.assertIsNotNone(self._config._pil_hist)

  def test_loop_applies_transform_before_recording_image_properties(
      self) -> None:
    """Tests that preview shape and dtype describe transformed images."""

    transformed = list()

    def transform(img: np.ndarray) -> np.ndarray:
      ret = np.flipud(img[:120, :160]).astype(np.uint16)
      transformed.append(ret)
      return ret

    self._config._transform = transform
    self.run_config_cycle(elapsed=0.5)

    self.assertEqual(len(transformed), 1)
    self.assertEqual(self._config.shape, (120, 160))
    self.assertEqual(self._config.dtype, 'uint16')
    bit_depth = int(np.ceil(np.log2(int(np.max(transformed[0])) + 1)))
    np.testing.assert_array_equal(self._config._original_img,
                                  (transformed[0] /
                                   2 ** (bit_depth - 8)).astype(np.uint8))
