# coding: utf-8

import logging
from multiprocessing import Event
from tkinter import TclError
from unittest.mock import patch
from crappy.blocks.stop_button import StopButton
import crappy.blocks.stop_button as stop_button_module

from ..block import BlockTestBase


class StopButtonTests:
  """Run the same clean-stop behavior against both graphical backends."""

  def setUp(self) -> None:
    """Tracks StopButton Blocks for cleanup, even after failed preparation."""

    super().setUp()
    patcher = patch.object(stop_button_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._buttons: list[StopButton] = list()

  def tearDown(self) -> None:
    """Closes windows before resetting the Block class state."""

    for button in self._buttons:
      button.finish()

    super().tearDown()

  def _prepare_stop_button(self, **kwargs) -> StopButton:
    """Creates a StopButton and prepares its GUI."""

    kwargs.setdefault('backend', self._backend)
    button = StopButton(**kwargs)
    self._buttons.append(button)
    button.prepare()
    if self._backend == 'tkinter':
      button._root.withdraw()
    return button

  def _click(self, button: StopButton) -> None:
    """Invoke the actual native widget callback."""

    if self._backend == 'tkinter':
      button._button.invoke()
    else:
      button._qt_button.click()

  @staticmethod
  def _capture_logs(button: StopButton) -> list[tuple[int, str]]:
    """Captures StopButton log calls without relying on logging handlers."""

    logs = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    button.log = log
    return logs

  def test_init_sets_block_options(self) -> None:
    """Checks StopButton-specific initialization."""

    button = StopButton(freq=None, display_freq=True, debug=True)

    self.assertIsNone(button.freq)
    self.assertTrue(button.display_freq)
    self.assertTrue(button.debug)
    self.assertFalse(button.pausable)
    self.assertIsNone(button._root)
    self.assertIsNone(button._label)
    self.assertIsNone(button._button)

  def test_prepare_creates_gui_without_links(self) -> None:
    """Checks the message above the native stop button."""

    button = self._prepare_stop_button()

    if self._backend == 'tkinter':
      self.assertEqual(button._root.title(), 'Stop Button Block')
      self.assertEqual(button._label.cget('text'),
                       'Click button to stop test')
      self.assertEqual(button._button.cget('text'), 'STOP')
      self.assertIsNone(button._qt_window)
    else:
      self.assertEqual(button._qt_window.windowTitle(), 'Stop Button Block')
      self.assertEqual(button._qt_label.text(), 'Click button to stop test')
      self.assertEqual(button._qt_button.text(), 'STOP')
      self.assertIsNone(button._root)
      layout = button._qt_window.layout()
      self.assertIs(layout.itemAt(0).widget(), button._qt_label)
      self.assertIs(layout.itemAt(1).layout().itemAt(0).widget(),
                     button._qt_button)
      constraints = stop_button_module.QtWidgets.QLayout.SizeConstraint
      self.assertEqual(layout.sizeConstraint(),
                       constraints.SetFixedSize)

  def test_loop_updates_gui(self) -> None:
    """Checks that loop services the selected GUI event queue."""

    button = self._prepare_stop_button()

    target = button._root if self._backend == 'tkinter' else button._qt_app
    method = 'update' if self._backend == 'tkinter' else 'processEvents'
    with patch.object(target, method) as update:
      button.loop()

    update.assert_called_once_with()

  def test_click_sets_stop_event(self) -> None:
    """Checks that clicking the GUI button triggers Block.stop."""

    button = self._prepare_stop_button()
    button._stop_event = Event()
    logs = self._capture_logs(button)

    self._click(button)

    self.assertTrue(button._stop_event.is_set())
    self.assertIn((logging.DEBUG, 'Button clicked in the GUI'), logs)
    self.assertIn((logging.WARNING,
                   'Stop button clicked, stopping the script !'), logs)
    self.assertIn((logging.WARNING,
                   'stop method called, setting the stop event !'), logs)

  def test_click_without_stop_event_is_safe(self) -> None:
    """Checks direct clicks before Block synchronization objects are set."""

    button = self._prepare_stop_button()
    logs = self._capture_logs(button)

    self._click(button)

    self.assertIsNone(button._stop_event)
    self.assertIn((logging.WARNING,
                   'Stop button clicked, stopping the script !'), logs)

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes this Block's window."""

    button = self._prepare_stop_button()

    button.finish()

    if self._backend == 'tkinter':
      with self.assertRaises(TclError):
        button._root.wm_state()
    else:
      self.assertFalse(button._qt_window.isVisible())

  def test_finish_is_safe_before_prepare(self) -> None:
    """Checks that finish accepts a StopButton without a window."""

    button = StopButton()

    button.finish()

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    button = self._prepare_stop_button()

    button.finish()
    button.finish()


class TestStopButton(StopButtonTests, BlockTestBase):
  """Legacy Tkinter stop-button integration."""

  _backend = 'tkinter'

  def test_loop_ignores_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    button = self._prepare_stop_button()
    with patch.object(button._root, 'update', side_effect=TclError):
      button.loop()


class TestStopButtonPyQt(StopButtonTests, BlockTestBase):
  """Native Qt stop requests, callback protection, and window lifecycle."""

  _backend = 'pyqt'

  def test_real_signal_failure_is_raised_by_loop(self) -> None:
    """A failing stop request cannot escape through a Qt signal handler."""

    button = self._prepare_stop_button()
    error = RuntimeError('stop failed')
    with patch.object(button, 'stop', side_effect=error):
      button._qt_button.click()
    self.assertIs(button._callback_error, error)
    self.assertFalse(button._qt_button.isEnabled())
    with self.assertRaisesRegex(RuntimeError, 'stop failed'):
      button.loop()

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
