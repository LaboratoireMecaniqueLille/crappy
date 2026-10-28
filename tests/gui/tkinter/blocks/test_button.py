from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch
from tests.support.block_test_case import BlockTestBase

from tests.gui._support.button import ButtonTests


class TestButton(ButtonTests, BlockTestBase):
  """Legacy Tkinter button integration."""

  _backend = 'tkinter'

  def test_loop_ignores_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    button, sink = self._prepare_button(spam=True)
    with patch.object(button._root, 'update', side_effect=TclError):
      button.loop()
    self.assertFalse(sink.inputs[0].poll())
