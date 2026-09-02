# coding: utf-8

import logging
from multiprocessing import Event
from tkinter import TclError
from unittest.mock import patch
from crappy.blocks.stop_button import StopButton

from ..block import BlockTestBase


class TestStopButton(BlockTestBase):
  """Unit tests for the StopButton Block-specific GUI behavior."""

  def setUp(self) -> None:
    """Tracks StopButton Blocks to destroy their Tk windows."""

    self._buttons: list[StopButton] = list()

  def tearDown(self) -> None:
    """Destroys Tk windows before resetting the Block class state."""

    for button in self._buttons:
      button.finish()

    super().tearDown()

  def _prepare_stop_button(self, **kwargs) -> StopButton:
    """Creates a StopButton and prepares its GUI."""

    button = StopButton(**kwargs)
    button.prepare()
    button._root.withdraw()

    self._buttons.append(button)
    return button

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
    """Checks the Tk window and widgets created by prepare."""

    button = self._prepare_stop_button()

    self.assertEqual(button._root.title(), 'Stop Button Block')
    self.assertEqual(button._label.cget('text'),
                     'Click button to stop test')
    self.assertEqual(button._button.cget('text'), 'STOP')

  def test_loop_updates_gui(self) -> None:
    """Checks that loop updates the Tk window."""

    button = self._prepare_stop_button()

    with patch.object(button._root, 'update') as update:
      button.loop()

    update.assert_called_once_with()

  def test_loop_ignores_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    button = self._prepare_stop_button()

    with patch.object(button._root, 'update', side_effect=TclError):
      button.loop()

  def test_click_sets_stop_event(self) -> None:
    """Checks that clicking the GUI button triggers Block.stop."""

    button = self._prepare_stop_button()
    button._stop_event = Event()
    logs = self._capture_logs(button)

    button._button.invoke()

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

    button._button.invoke()

    self.assertIsNone(button._stop_event)
    self.assertIn((logging.WARNING,
                   'Stop button clicked, stopping the script !'), logs)

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes the Tk root."""

    button = self._prepare_stop_button()

    button.finish()

    with self.assertRaises(TclError):
      button._root.wm_state()

  def test_finish_is_safe_before_prepare(self) -> None:
    """Checks that finish accepts a StopButton without a Tk root."""

    button = StopButton()

    button.finish()

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    button = self._prepare_stop_button()

    button.finish()
    button.finish()
