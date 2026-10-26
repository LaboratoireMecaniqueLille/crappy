from tests.support.gui import require_gui_group

require_gui_group('tkinter')

from tkinter import TclError
from unittest.mock import patch
import crappy.blocks.canvas as canvas_module
from tests.support.blocks import BlockTestBase

from tests.gui._support.canvas import CanvasTests


class TestCanvas(CanvasTests, BlockTestBase):
  """Legacy Canvas embeds a Matplotlib figure in its own Tk window."""

  _backend = 'tkinter'

  def test_prepare_embeds_figure_and_color_bar(self) -> None:
    """The Tk canvas uses a FigureCanvasTkAgg rather than a pyplot window."""

    for dpi in (100, 101.6):
      with self.subTest(dpi=dpi):
        with (patch.dict(canvas_module.mpl.rcParams, {'figure.dpi': dpi}),
              patch.object(canvas_module.mpl_figure, 'Figure',
                           wraps=canvas_module.mpl_figure.Figure) as figure):
          canvas, _ = self._prepare_canvas(title='Test Canvas',
                                           window_size=(3, 2),
                                           color_range=(1, 5))
        # Check the requested size before Tk applies display scaling and
        # rounds the widget dimensions, possibly more than once
        figure.assert_called_once_with(figsize=(3, 2))
        self.assertEqual(canvas._root.title(), 'Test Canvas')
        self.assertIs(canvas._tk_canvas.figure, canvas._fig)
        self.assertIs(canvas._tk_canvas.get_tk_widget().master, canvas._root)
        self.assertEqual(canvas.ax.get_title(), 'Test Canvas')
        self.assertFalse(canvas.ax.axison)
        self.assertEqual(len(canvas._fig.axes), 2)
        self.assertEqual(canvas._fig.axes[1].get_xlabel(), 'Dot text values')
        self.assertEqual([text.get_text() for text in
                          canvas._fig.axes[1].get_xticklabels()], ['1', '5'])
        self.assertIsNone(canvas._qt_window)

  def test_loop_services_idle_gui_without_redrawing(self) -> None:
    """Without data or a timer, pump Tk events but avoid unnecessary draws."""

    canvas, _ = self._prepare_canvas(draw=self._draw[:1])
    with (patch.object(canvas._fig.canvas, 'flush_events') as flush,
          patch.object(canvas._fig.canvas, 'draw') as draw):
      canvas.loop()
    flush.assert_called_once_with()
    draw.assert_not_called()

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
