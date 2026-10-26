from tests.support.gui import require_gui_group

require_gui_group('pyqt')

from tests.support.blocks import exception_leaves
from unittest.mock import patch
import crappy.blocks.button as button_module
from tests.support.blocks import BlockTestBase

from tests.gui._support.button import ButtonTests


class TestButtonPyQt(ButtonTests, BlockTestBase):
  """Native Qt button signals, rendering, and window lifecycle."""

  _backend = 'pyqt'

  def test_real_signal_failure_is_raised_by_loop(self) -> None:
    """A failing send must be retained rather than escape the Qt signal."""

    button, _ = self._prepare_button()
    error = RuntimeError('send failed')
    with patch.object(button, 'send', side_effect=error):
      button._qt_button.click()
    self.assertIs(button._failure_manager.snapshot()[0][1], error)
    self.assertFalse(button._qt_button.isEnabled())
    with self.assertRaises(ExceptionGroup) as caught:
      button.loop()
    self.assertEqual(exception_leaves(caught.exception), (error,))
    self.assertFalse(hasattr(button, '_callback_error'))

  def test_closing_window_does_not_stop_spam_or_other_windows(self) -> None:
    """Closing this window is not a request to stop the running test."""

    button, sink = self._prepare_button(spam=True)
    other, _ = self._prepare_button()
    self.assertIs(button._qt_app, other._qt_app)
    with patch.object(button, 'stop') as stop:
      button._qt_window.close()
      with patch.object(button_module, 'time', return_value=14.0):
        button.loop()
      stop.assert_not_called()
    self.assertEqual(sink.inputs[0].recv(), {'t(s)': 4.0, 'step': 0})
    button.finish()
    self.assertTrue(other._qt_window.isVisible())

  def test_label_uses_plain_text(self) -> None:
    """Label names must not be interpreted as HTML by Qt."""

    button, _ = self._prepare_button(label='<b>step</b>')
    self.assertEqual(button._qt_label.text(), '<b>step</b>: 0')
    self.assertEqual(button._qt_label.textFormat(),
                     button_module.QtCore.Qt.TextFormat.PlainText)
