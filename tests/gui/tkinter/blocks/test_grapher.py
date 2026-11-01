from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch

from tests.gui._support.grapher import GrapherTestBase


class TestGrapher(GrapherTestBase):
  def test_loop_ignores_tcl_errors_while_drawing(self) -> None:
    """Checks that draw and event flushing errors are tolerated."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=10)

    def recv_all_data_raw():
      return [{'x': [1], 'y': [2]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    canvas = grapher._canvas
    if canvas is None:
      self.fail("The Grapher canvas was not initialized")

    with patch.object(canvas, 'flush_events', side_effect=TclError):
      grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [2])
