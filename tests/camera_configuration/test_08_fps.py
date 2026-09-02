# coding: utf-8

from unittest.mock import patch

import crappy.tool.camera_config.camera_config as camera_config_module

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestFPS(ConfigurationWindowTestBase):
  """Class for testing if the FPS are correctly handled in the configuration 
  window.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def test_fps(self) -> None:
    """Tests whether the FPS are correctly calculated, displayed, and if the
    maximum FPs value is enforced."""

    # FPS-related variables should be initialized to their default values
    self.assertEqual(self._config._fps_var.get(), 0.)
    self.assertEqual(self._config._fps_txt.get(),
                     f'fps = 0.00\n(might be lower in this GUI than actual)')

    current_time = [0.0]
    self._config._last_upd_t = current_time[0]

    def fake_time() -> float:
      return current_time[0]

    def acquire_image() -> None:
      self._config._n_loops += 1

    with patch.object(camera_config_module, 'time', side_effect=fake_time), \
         patch.object(self._config, '_update_img', side_effect=acquire_image):
      # Ten evenly-spaced frames are enough to verify each frequency exactly;
      # there is no need to wait through the equivalent wall-clock duration.
      for fps in (1, 2, 3, 4, 5, 10, 15, 20):
        with self.subTest(fps=fps):
          self._config._max_freq = fps

          for _ in range(10):
            current_time[0] += 1 / fps
            self._config._img_acq_sched()
          self._config._upd_var_sched()

          self.assertAlmostEqual(self._config._fps_var.get(), fps)
          self.assertEqual(
              self._config._fps_txt.get(),
              f'fps = {self._config._fps_var.get():.2f}\n(might be lower in '
              f'this GUI than actual)')

      # Free-looping should not impose the configured 20 FPS ceiling.
      self._config._max_freq = None
      for _ in range(25):
        current_time[0] += 1 / 25
        self._config._img_acq_sched()
      self._config._upd_var_sched()

    self.assertAlmostEqual(self._config._fps_var.get(), 25.0)
