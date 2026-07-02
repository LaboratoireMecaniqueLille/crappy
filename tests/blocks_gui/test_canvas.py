# coding: utf-8

from multiprocessing import Value
from unittest.mock import patch
from tkinter import TclError
import crappy
from crappy.blocks.canvas import Canvas, DotText, Text, Time
import crappy.blocks.canvas as canvas_module

from ..block import BlockTestBase, TestBlock, link


class TestCanvas(BlockTestBase):
  """Unit tests for the Canvas Block-specific GUI behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Tracks the Canvas Blocks and closes stale Matplotlib figures."""

    self._canvases: list[Canvas] = list()
    self._image = str(crappy.resources.paths['pad'])
    canvas_module.plt.close('all')

  def tearDown(self) -> None:
    """Closes Matplotlib figures before resetting the Block class state."""

    for canvas in self._canvases:
      canvas.finish()
    canvas_module.plt.close('all')

    super().tearDown()

  def _prepare_canvas(self, **kwargs) -> tuple[Canvas, TestBlock]:
    """Creates a linked Canvas and prepares its Matplotlib figure."""

    kwargs.setdefault('image_path', self._image)
    kwargs.setdefault('backend', 'Agg')

    source = TestBlock()
    canvas = Canvas(**kwargs)
    canvas._instance_t0 = Value('d', self._t0)
    link(source, canvas)

    with (patch.object(canvas_module.plt, 'show'),
          patch.object(canvas_module.plt, 'pause')):
      canvas.prepare()

    self._canvases.append(canvas)
    return canvas, source

  def test_arguments_are_validated(self) -> None:
    """Checks early validation of draw dictionaries and color ranges."""

    with self.assertRaises(ValueError):
      Canvas(self._image, draw=[{'coord': (0, 0)}], backend='Agg')

    with self.assertRaises(ValueError):
      Canvas(self._image,
             draw=[{'type': 'unknown', 'coord': (0, 0)}],
             backend='Agg')

    with self.assertRaises(ValueError):
      Canvas(self._image, color_range=(1, 1), backend='Agg')

    canvas = Canvas(self._image, color_range=(5, 1), backend='Agg')

    self.assertEqual(canvas.color_range, (1, 5))

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Canvas without input Links fails early."""

    canvas = Canvas(self._image, backend='Agg')

    with self.assertRaises(IOError):
      canvas.prepare()

  def test_prepare_builds_figure_and_drawing_elements(self) -> None:
    """Checks the Matplotlib figure and drawable elements created by prepare."""

    draw = [
      {'type': 'text', 'coord': (10, 20), 'text': 'T = %.1f',
       'label': 'temperature'},
      {'type': 'dot_text', 'coord': (30, 40), 'text': 'P = %.2f',
       'label': 'pressure'},
      {'type': 'time', 'coord': (50, 60), 'text': '', 'label': ''},
    ]

    canvas, _ = self._prepare_canvas(draw=draw,
                                     color_range=(1, 5),
                                     title='Test Canvas',
                                     window_size=(3, 2))

    self.assertIsNotNone(canvas._fig)
    self.assertIsNotNone(canvas.ax)
    self.assertEqual(canvas.ax.get_title(), 'Test Canvas')
    self.assertFalse(canvas.ax.axison)
    self.assertEqual(len(canvas._drawing_elements), 3)
    self.assertIsInstance(canvas._drawing_elements[0], Text)
    self.assertIsInstance(canvas._drawing_elements[1], DotText)
    self.assertIsInstance(canvas._drawing_elements[2], Time)
    self.assertEqual(canvas._drawing_elements[0]._txt.get_text(), 'T = %.1f')
    self.assertEqual(canvas._drawing_elements[1]._txt.get_text(), 'P = %.2f')
    self.assertEqual(canvas._drawing_elements[2]._txt.get_text(), '00:00')
    self.assertEqual(len(canvas._fig.axes), 2)
    self.assertEqual(canvas._fig.axes[1].get_xlabel(), 'Dot text values')

  def test_loop_updates_text_dot_and_time_from_received_data(self) -> None:
    """Checks the drawable updates driven by the latest incoming payload."""

    draw = [
      {'type': 'text', 'coord': (10, 20), 'text': 'T = %.1f',
       'label': 'temperature'},
      {'type': 'dot_text', 'coord': (30, 40), 'text': 'P = %.2f',
       'label': 'pressure'},
      {'type': 'time', 'coord': (50, 60), 'text': '', 'label': ''},
    ]
    canvas, source = self._prepare_canvas(draw=draw, color_range=(1, 5))

    source.send({'temperature': 3.5, 'pressure': 4.0})

    with (patch.object(canvas_module, 'time', return_value=17.0),
          patch.object(canvas_module.plt, 'pause') as pause):
      canvas.loop()

    text, dot_text, time_text = canvas._drawing_elements
    self.assertEqual(text._txt.get_text(), 'T = 3.5')
    self.assertEqual(dot_text._txt.get_text(), 'P = 4.00')
    self.assertEqual(time_text._txt.get_text(), '0:00:07')
    self.assertEqual(dot_text._dot.get_facecolor(),
                     canvas_module.mpl.cm.coolwarm(0.75))
    pause.assert_called_once_with(0.001)

  def test_loop_does_not_update_without_new_data(self) -> None:
    """Checks that loop stays quiet when no incoming payload is available."""

    canvas, _ = self._prepare_canvas(
      draw=[{'type': 'time', 'coord': (50, 60), 'text': '', 'label': ''}])

    with (patch.object(canvas_module, 'time', return_value=17.0),
          patch.object(canvas_module.plt, 'pause') as pause):
      canvas.loop()

    self.assertEqual(canvas._drawing_elements[0]._txt.get_text(), '00:00')
    pause.assert_not_called()

  def test_loop_ignores_tcl_errors_while_drawing(self) -> None:
    """Checks that a closed Tk-backed Matplotlib window does not raise."""

    canvas, source = self._prepare_canvas(
      draw=[{'type': 'text', 'coord': (10, 20), 'text': 'T = %.1f',
             'label': 'temperature'}])
    source.send({'temperature': 3.5})

    with (patch.object(canvas._fig.canvas, 'draw', side_effect=TclError),
          patch.object(canvas_module.plt, 'pause') as pause):
      canvas.loop()

    self.assertEqual(canvas._drawing_elements[0]._txt.get_text(), 'T = 3.5')
    pause.assert_called_once_with(0.001)

  def test_finish_closes_only_its_own_figure(self) -> None:
    """Checks that finish does not close another current Matplotlib figure."""

    canvas_1, _ = self._prepare_canvas(title='first')
    canvas_2, _ = self._prepare_canvas(title='second')
    canvas_module.plt.figure(canvas_2._fig.number)

    canvas_1.finish()

    self.assertNotIn(canvas_1._fig.number, canvas_module.plt.get_fignums())
    self.assertIn(canvas_2._fig.number, canvas_module.plt.get_fignums())

  def test_finish_is_noop_before_prepare(self) -> None:
    """Checks that finish accepts a Canvas whose figure was never created."""

    canvas = Canvas(self._image, backend='Agg')

    canvas.finish()
