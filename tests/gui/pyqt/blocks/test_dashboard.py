from tests.support.gui import require_gui_group

require_gui_group('pyqt')

from unittest.mock import patch
import crappy.blocks.dashboard as dashboard_module
from tests.support.block_test_case import BlockTestBase

from tests.gui._support.dashboard import DashboardTests


class TestDashboardPyQt(DashboardTests, BlockTestBase):
  """Native Qt Dashboard layout, rendering, and application reuse."""

  _backend = 'pyqt'

  def test_labels_and_received_strings_use_plain_text(self) -> None:
    """Names and values resembling HTML must be displayed literally."""

    label = '<b>value</b>'
    dashboard, source = self._prepare_dashboard((label,))
    source.send({label: '<i>text</i>'})
    dashboard.loop()
    self.assertEqual(self._value(dashboard, label), '<i>text</i>')
    for widget in (dashboard._qt_labels[label], dashboard._qt_values[label]):
      self.assertEqual(widget.textFormat(),
                       dashboard_module.QtCore.Qt.TextFormat.PlainText)

  def test_closing_window_does_not_stop_test_or_other_windows(self) -> None:
    """Closing one Dashboard leaves the application and its peers alive."""

    dashboard, _ = self._prepare_dashboard(('a',))
    other, _ = self._prepare_dashboard(('b',))
    self.assertIs(dashboard._qt_app, other._qt_app)
    with patch.object(dashboard, 'stop') as stop:
      dashboard._qt_window.close()
      dashboard.loop()
      stop.assert_not_called()
    dashboard.finish()
    self.assertTrue(other._qt_window.isVisible())
