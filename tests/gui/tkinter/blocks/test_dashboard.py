from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch
from tests.support.block_test_case import BlockTestBase

from tests.gui._support.dashboard import DashboardTests


class TestDashboard(DashboardTests, BlockTestBase):
  """Legacy Tkinter Dashboard integration."""

  _backend = 'tkinter'

  def test_loop_ignores_update_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    dashboard, source = self._prepare_dashboard(('a',))
    source.send({'a': 1})
    with patch.object(dashboard._dashboard, 'update', side_effect=TclError):
      dashboard.loop()
    self.assertEqual(self._value(dashboard, 'a'), '1.00')
