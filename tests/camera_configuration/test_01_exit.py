# coding: utf-8

from tkinter import TclError

from .camera_configuration_test_base import ConfigurationWindowTestBase


class TestFinish(ConfigurationWindowTestBase):
  """"""

  def test_exit(self) -> None:
    """"""

    # Destroying the main window
    self._config.finish()

    # This call should raise an error as the window shouldn't exist anymore
    with self.assertRaises(TclError):
      self._config.wm_state()

    self.assertFalse(self._config._run)
    self.assertFalse(self._config._histogram_process.is_alive())

    # Indicating the tearDown() method not to destroy the window
    self._exit = False
