# coding: utf-8

from time import time, sleep

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

    # Testing a range of realistic frequencies
    for fps in (1, 2, 3, 4, 5, 10, 15, 20):
      with self.subTest(fps=fps):

        # Setting the maximum frequency in the interface
        self._config._max_freq = fps

        # Continuously updating the image
        t0 = time()
        while time() - t0 < 10 / fps:
          self._config._img_acq_sched()
          sleep(0.001)
        # Refreshing the indicators
        self._config._upd_var_sched()

        # Checking that the frequency is less than 20% off compared to target
        self.assertAlmostEqual(self._config._fps_var.get(), fps,
                               delta=fps * 0.2)
        self.assertEqual(self._config._fps_txt.get(),
                         f'fps = {self._config._fps_var.get():.2f}\n(might be '
                         f'lower in this GUI than actual)')

    # Same test but this time with free-looping
    self._config._max_freq = None
    t0 = time()
    while time() - t0 < 2:
      self._config._img_acq_sched()
      sleep(0.001)
    self._config._upd_var_sched()
    self.assertGreater(self._config._fps_var.get(), 20)
