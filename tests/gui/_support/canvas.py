from tests.support.blocks import with_failure_manager
from multiprocessing import Value
from unittest.mock import patch
import crappy
from crappy.blocks.canvas import Canvas, DotText, Text, Time
import crappy.blocks.canvas as canvas_module
from tests.support.blocks import TestBlock, link


class CanvasTests:
  """Exercise equivalent overlays and updates on the two GUI backends."""

  _t0 = 10.0
  _draw = (
    {'type': 'text', 'coord': (10, 20), 'text': 'T = %.1f',
     'label': 'temperature'},
    {'type': 'dot_text', 'coord': (30, 40), 'text': 'P = %.2f',
     'label': 'pressure'},
    {'type': 'time', 'coord': (50, 60)},
  )

  def setUp(self) -> None:
    """Track windows before preparation so failures also get cleaned up."""

    super().setUp()
    patcher = patch.object(canvas_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._canvases: list[Canvas] = list()
    self._image = crappy.resources.paths['pad']

  def tearDown(self) -> None:
    """Close both native and Matplotlib-backed windows before Block.reset."""

    for canvas in self._canvases:
      canvas.finish()
    super().tearDown()

  def _prepare_canvas(self, **kwargs) -> tuple[Canvas, TestBlock]:
    """Create a linked Canvas using the backend selected by the test class."""

    kwargs.setdefault('image_path', self._image)
    kwargs.setdefault('backend', self._backend)
    source = TestBlock()
    canvas = with_failure_manager(Canvas(**kwargs))
    canvas._instance_t0 = Value('d', self._t0)
    link(source, canvas)
    self._canvases.append(canvas)
    canvas.prepare()
    if self._backend == 'tkinter':
      canvas._root.withdraw()
    return canvas, source

  def _text(self, element: Text) -> str:
    """Read the text from the native graphical item for this backend."""

    return (element._txt.get_text() if self._backend == 'tkinter'
            else element._qt_txt.text())

  def test_prepare_creates_all_drawing_elements(self) -> None:
    """Text, dot-text, and time overlays start with equivalent content."""

    canvas, _ = self._prepare_canvas(draw=self._draw, color_range=(1, 5))
    self.assertEqual(len(canvas._drawing_elements), 3)
    for element, kind, text in zip(canvas._drawing_elements,
                                    (Text, DotText, Time),
                                    ('T = %.1f', 'P = %.2f', '00:00')):
      self.assertIsInstance(element, kind)
      self.assertEqual(self._text(element), text)

  def test_loop_updates_text_dot_and_time_from_latest_data(self) -> None:
    """Multiple incoming messages update the overlays with the last values."""

    canvas, source = self._prepare_canvas(draw=self._draw, color_range=(1, 5))
    source.send({'temperature': 1.0, 'pressure': 2.0})
    source.send({'temperature': 3.5, 'pressure': 4.0})
    with patch.object(canvas_module, 'time', return_value=17.0):
      canvas.loop()
    text, dot_text, time_text = canvas._drawing_elements
    self.assertEqual(self._text(text), 'T = 3.5')
    self.assertEqual(self._text(dot_text), 'P = 4.00')
    self.assertEqual(self._text(time_text), '0:00:07')
    if self._backend == 'tkinter':
      self.assertEqual(dot_text._dot.get_facecolor(),
                       canvas_module.mpl.cm.coolwarm(0.75))
    else:
      self.assertEqual(dot_text._qt_dot.brush().color().getRgb(),
                       (244, 152, 122, 255))

  def test_time_advances_without_incoming_data(self) -> None:
    """The clock must advance even while every upstream Block is idle."""

    canvas, _ = self._prepare_canvas(draw=[self._draw[2]])
    with patch.object(canvas_module, 'time', return_value=17.0):
      canvas.loop()
    self.assertEqual(self._text(canvas._drawing_elements[0]), '0:00:07')

  def test_missing_labels_leave_text_and_dot_unchanged(self) -> None:
    """Unrelated or empty payloads do not reset already displayed values."""

    canvas, source = self._prepare_canvas(draw=self._draw[:2])
    source.send({'temperature': 3.5})
    canvas.loop()
    source.send({'unrelated': 4.0})
    canvas.loop()
    canvas.loop()
    self.assertEqual(self._text(canvas._drawing_elements[0]), 'T = 3.5')
    self.assertEqual(self._text(canvas._drawing_elements[1]), 'P = %.2f')

  def test_finish_closes_only_its_own_window(self) -> None:
    """Finishing one Canvas leaves the other window available."""

    canvas, _ = self._prepare_canvas(title='first')
    other, _ = self._prepare_canvas(title='second')
    window = (canvas._root if self._backend == 'tkinter'
              else canvas._qt_window)
    canvas.finish()
    canvas.finish()
    if self._backend == 'tkinter':
      from tkinter import TclError
      self.assertIsNone(canvas._root)
      with self.assertRaises(TclError):
        window.wm_state()
      self.assertEqual(other._root.wm_state(), 'withdrawn')
    else:
      self.assertIsNone(canvas._qt_window)
      self.assertFalse(window.isVisible())
      self.assertTrue(other._qt_window.isVisible())
      self.assertIs(canvas._qt_app, other._qt_app)
