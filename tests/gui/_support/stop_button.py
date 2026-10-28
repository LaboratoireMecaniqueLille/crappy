from tests.support.assertions import with_failure_manager
import logging
from multiprocessing import Event
from unittest.mock import patch
from crappy.blocks.stop_button import StopButton
import crappy.blocks.stop_button as stop_button_module


class StopButtonTests:
  """Run the same clean-stop behavior against both graphical backends."""

  def setUp(self) -> None:
    """Tracks StopButton Blocks for cleanup, even after failed preparation."""

    super().setUp()
    patcher = patch.object(stop_button_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._buttons: list[StopButton] = list()

  def _prepare_stop_button(self, **kwargs) -> StopButton:
    """Creates a StopButton and prepares its GUI."""

    kwargs.setdefault('backend', self._backend)
    button = with_failure_manager(StopButton(**kwargs))
    self.addCleanup(button.finish)
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

    button = with_failure_manager(StopButton(freq=None, display_freq=True, debug=True))
    self.addCleanup(button.finish)

    self.assertIsNone(button.freq)
    self.assertTrue(button.display_freq)
    self.assertTrue(button.debug)
    self.assertFalse(button.pausable)

  def test_prepare_creates_gui_without_links(self) -> None:
    """A standalone stop button can receive a click and stop its run."""

    button = self._prepare_stop_button()
    button._stop_event = Event()
    self._click(button)
    self.assertTrue(button._stop_event.is_set())
    if self._backend == 'tkinter':
      self.assertTrue(button._root.winfo_exists())
    else:
      self.assertTrue(button._qt_window.isVisible())

  def test_loop_updates_gui(self) -> None:
    """The control processes queued UI work while waiting for a click."""

    button = self._prepare_stop_button()
    handled = []
    if self._backend == 'tkinter':
      button._root.after_idle(lambda: handled.append(True))
    else:
      from PyQt6.QtCore import QTimer
      QTimer.singleShot(0, lambda: handled.append(True))
    button.loop()
    self.assertEqual(handled, [True])

  def test_click_sets_stop_event(self) -> None:
    """Checks that clicking the GUI button triggers Block.stop."""

    button = self._prepare_stop_button()
    button._stop_event = Event()
    logs = self._capture_logs(button)

    self._click(button)

    self.assertTrue(button._stop_event.is_set())
    self.assertTrue(any(level >= logging.WARNING for level, _ in logs))

  def test_click_without_stop_event_is_safe(self) -> None:
    """Checks direct clicks before Block synchronization objects are set."""

    button = self._prepare_stop_button()
    logs = self._capture_logs(button)

    self._click(button)

    button._stop_event = Event()
    self._click(button)
    self.assertTrue(button._stop_event.is_set())
    self.assertTrue(any(level >= logging.WARNING for level, _ in logs))

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes this Block's window."""

    button = self._prepare_stop_button()
    window = (button._root if self._backend == 'tkinter'
              else button._qt_window)

    button.finish()

    if self._backend == 'tkinter':
      from tkinter import TclError
      with self.assertRaises(TclError):
        window.wm_state()
    else:
      self.assertFalse(window.isVisible())

  def test_finish_is_safe_before_prepare(self) -> None:
    """Checks that finish accepts a StopButton without a window."""

    button = with_failure_manager(StopButton())
    self.addCleanup(button.finish)

    button.finish()

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    button = self._prepare_stop_button()

    button.finish()
    button.finish()
