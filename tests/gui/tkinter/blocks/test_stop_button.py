from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch
from tests.support.block_test_case import BlockTestBase

from tests.gui._support.stop_button import StopButtonTests


class TestStopButton(StopButtonTests, BlockTestBase):
  """Legacy Tkinter stop-button integration."""

  _backend = 'tkinter'

  def test_loop_ignores_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    button = self._prepare_stop_button()
    with patch.object(button._root, 'update', side_effect=TclError):
      button.loop()
