# coding: utf-8

from .camera_configuration_test_base import ConfigurationWindowTestBase, tk


class TestFinish(ConfigurationWindowTestBase):
  """Class for testing the exit behavior of the configuration window.

  .. versionadded:: 2.0.8
  """

  start_histogram_process = True

  def test_exit(self) -> None:
    """Tests whether the configuration window exits as expected when closed."""

    # The stop event should not be set
    self.assertFalse(self._config._stop_event.is_set())

    # The histogram process should still be alive
    self.assertTrue(self._config._histogram_process.is_alive())

    # Destroying the main window
    self._config.finish()
    self._config._histogram_process.join(1.0)

    # The stop event should be set
    self.assertTrue(self._config._stop_event.is_set())

    # This call should raise an error as the window shouldn't exist anymore
    with self.assertRaises(tk.TclError):
      self._config.wm_state()

    # The histogram process should have been killed
    self.assertFalse(self._config._histogram_process.is_alive())
