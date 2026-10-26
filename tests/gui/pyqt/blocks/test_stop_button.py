from tests.support.gui import require_gui_group

require_gui_group('pyqt')

from tests.support.blocks import exception_leaves
from multiprocessing import Event
from unittest.mock import patch
from tests.support.blocks import BlockTestBase

from tests.gui._support.stop_button import StopButtonTests


class TestStopButtonPyQt(StopButtonTests, BlockTestBase):
  """Native Qt stop requests, callback protection, and window lifecycle."""

  _backend = 'pyqt'

  def test_real_signal_failure_is_raised_by_loop(self) -> None:
    """A failing stop request cannot escape through a Qt signal handler."""

    button = self._prepare_stop_button()
    error = RuntimeError('stop failed')
    with patch.object(button, 'stop', side_effect=error):
      button._qt_button.click()
    self.assertIs(button._failure_manager.snapshot()[0][1], error)
    self.assertFalse(button._qt_button.isEnabled())
    with self.assertRaises(ExceptionGroup) as caught:
      button.loop()
    self.assertEqual(exception_leaves(caught.exception), (error,))
    self.assertFalse(hasattr(button, '_callback_error'))

  def test_closing_window_is_not_a_stop_request(self) -> None:
    """Only the STOP button, not the window close control, stops the test."""

    button = self._prepare_stop_button()
    button._stop_event = Event()
    other = self._prepare_stop_button()
    self.assertIs(button._qt_app, other._qt_app)
    button._qt_window.close()
    button.loop()
    self.assertFalse(button._stop_event.is_set())
    button.finish()
    self.assertTrue(other._qt_window.isVisible())
