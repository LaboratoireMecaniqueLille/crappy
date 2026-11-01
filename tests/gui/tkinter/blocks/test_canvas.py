from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch
import crappy.blocks.canvas as canvas_module
from tests.support.block_test_case import BlockTestBase

from tests.gui._support.canvas import CanvasTests


class TestCanvas(CanvasTests, BlockTestBase):
  """Legacy Canvas embeds a Matplotlib figure in its own Tk window."""

  _backend = 'tkinter'

  def test_prepare_embeds_figure_and_color_bar(self) -> None:
    """The requested figure size, title, and color range appear in the window."""

    for dpi in (100, 101.6):
      with self.subTest(dpi=dpi):
        with patch.dict(canvas_module.mpl.rcParams, {'figure.dpi': dpi}):
          canvas, _ = self._prepare_canvas(title='Test Canvas',
                                           window_size=(3, 2), color_range=(1, 5))
        self.assertTrue(canvas._root.winfo_exists())
        self.assertEqual(canvas._root.title(), 'Test Canvas')
        self.assertEqual(canvas.ax.get_title(), 'Test Canvas')
        # Tk rounds physical dimensions to whole pixels during embedding.
        for actual, requested in zip(canvas.ax.figure.get_size_inches(), (3, 2)):
          self.assertAlmostEqual(actual, requested,
                                 delta=2 / canvas.ax.figure.dpi)
        legends = [axis for axis in canvas.ax.figure.axes if axis is not canvas.ax]
        displayed_values = []
        for axis in legends:
          for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            try:
              displayed_values.append(float(label.get_text()))
            except ValueError:
              pass
        self.assertIn(1, displayed_values)
        self.assertIn(5, displayed_values)

  def test_loop_services_idle_gui_without_new_data(self) -> None:
    """An idle Canvas processes deferred work and retains its displayed data."""

    canvas, source = self._prepare_canvas(draw=self._draw[:1])
    source.send({'temperature': 3.5})
    canvas.loop()
    before = self._text(canvas._drawing_elements[0])
    handled = []
    canvas._root.after_idle(lambda: handled.append(True))
    canvas.loop()
    self.assertEqual(handled, [True])
    self.assertEqual(self._text(canvas._drawing_elements[0]), before)

  def test_loop_ignores_tcl_errors_while_drawing(self) -> None:
    """A closed Tk window must not let a redraw error escape."""

    canvas, source = self._prepare_canvas(draw=self._draw[:1])
    source.send({'temperature': 3.5})
    with patch.object(canvas._fig.canvas, 'draw', side_effect=TclError):
      canvas.loop()
    self.assertEqual(self._text(canvas._drawing_elements[0]), 'T = 3.5')

  def test_loop_returns_when_tk_event_processing_fails(self) -> None:
    """Do not consume data when the Tk window has already disappeared."""

    canvas, source = self._prepare_canvas(draw=self._draw[:1])
    source.send({'temperature': 3.5})
    with patch.object(canvas._fig.canvas, 'flush_events', side_effect=TclError):
      canvas.loop()
    self.assertEqual(self._text(canvas._drawing_elements[0]), 'T = %.1f')
