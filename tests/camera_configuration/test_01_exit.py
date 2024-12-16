# coding: utf-8

from tkinter import TclError

from .camera_configuration_test_base import ConfigurationWindowTestBase


class TestFinish(ConfigurationWindowTestBase):
  """"""

  def test_exit(self) -> None:
    """"""

    # The stop event should not be set
    self.assertFalse(self._config._stop_event.is_set())

    # The histogram process should still be alive
    self.assertTrue(self._config._histogram_process.is_alive())

    # Destroying the main window
    self._config.finish()

    # The stop event should be set
    self.assertTrue(self._config._stop_event.is_set())

    # This call should raise an error as the window shouldn't exist anymore
    with self.assertRaises(TclError):
      self._config.wm_state()

    # The histogram process should have been killed
    self.assertFalse(self._config._histogram_process.is_alive())

    # Indicating the tearDown() method not to destroy the window
    self._exit = False
