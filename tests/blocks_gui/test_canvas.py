# coding: utf-8

from multiprocessing import Value
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import TclError
from unittest.mock import patch, sentinel
import crappy
from crappy.blocks.canvas import Canvas, DotText, Text, Time
import crappy.blocks.canvas as canvas_module

from ..block import BlockTestBase, TestBlock, link


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
    canvas = Canvas(**kwargs)
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
      self.assertIsNone(canvas._root)
      with self.assertRaises(TclError):
        window.wm_state()
      self.assertEqual(other._root.wm_state(), 'withdrawn')
    else:
      self.assertIsNone(canvas._qt_window)
      self.assertFalse(window.isVisible())
      self.assertTrue(other._qt_window.isVisible())
      self.assertIs(canvas._qt_app, other._qt_app)


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


class TestCanvasPyQt(CanvasTests, BlockTestBase):
  """Native Qt scene, image rendering, overlays, and viewport fitting."""

  _backend = 'pyqt'

  def test_entire_canvas_is_native_qt_without_matplotlib_or_tk(self) -> None:
    """The image, overlays, title, and color bar belong to one Qt window."""

    with (patch.object(canvas_module, 'mpl', sentinel.unused_mpl),
          patch.object(canvas_module, 'mpl_figure', sentinel.unused_figure),
          patch.object(canvas_module, 'mpl_image', sentinel.unused_image),
          patch.object(canvas_module, 'mpl_patches', sentinel.unused_patches),
          patch.object(canvas_module, 'backend_tkagg', sentinel.unused_tkagg),
          patch.object(canvas_module.tk, 'Tk',
                       side_effect=AssertionError('Tk used'))):
      canvas, source = self._prepare_canvas(draw=self._draw, color_range=(1, 5),
                                            title='<b>Test Canvas</b>')
      source.send({'temperature': 3.5, 'pressure': 4.0})
      canvas.loop()

    widgets = canvas_module.QtWidgets
    self.assertIsInstance(canvas._qt_window, widgets.QMainWindow)
    self.assertIsInstance(canvas._qt_view, widgets.QGraphicsView)
    self.assertIs(canvas._qt_view.scene(), canvas.qt_scene)
    self.assertIs(canvas.qt_scene.parent(), canvas._qt_window)
    panel = canvas._qt_window.centralWidget()
    self.assertIs(canvas._qt_view.parent(), panel)
    self.assertEqual(canvas._qt_window.windowTitle(), '<b>Test Canvas</b>')
    self.assertIsNone(canvas._fig)
    self.assertIsNone(canvas.ax)
    self.assertIsNone(canvas._root)
    labels = canvas._qt_window.findChildren(widgets.QLabel)
    texts = [label.text() for label in labels]
    for text in ('<b>Test Canvas</b>', 'Dot text values', '1', '5'):
      self.assertIn(text, texts)
    title = next(label for label in labels
                 if label.text() == '<b>Test Canvas</b>')
    self.assertEqual(title.textFormat(),
                     canvas_module.QtCore.Qt.TextFormat.PlainText)
    bars = [label for label in labels if not label.pixmap().isNull()]
    self.assertEqual(len(bars), 1)
    self.assertEqual((bars[0].pixmap().width(), bars[0].pixmap().height()),
                     (256, 16))
    backgrounds = [item for item in canvas.qt_scene.items()
                   if isinstance(item, widgets.QGraphicsPixmapItem)]
    self.assertEqual(len(backgrounds), 1)
    background = backgrounds[0]
    image = canvas_module.QtGui.QImage(str(self._image))
    self.assertEqual(background.pixmap().size(), image.size())
    self.assertEqual(canvas.qt_scene.sceneRect(),
                     canvas_module.QtCore.QRectF(image.rect()))
    self.assertFalse(canvas._qt_window.findChildren(widgets.QToolBar))
    self.assertEqual(canvas._qt_view.dragMode(),
                     widgets.QGraphicsView.DragMode.NoDrag)
    policy = canvas_module.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    self.assertEqual(canvas._qt_view.horizontalScrollBarPolicy(), policy)
    self.assertEqual(canvas._qt_view.verticalScrollBarPolicy(), policy)

  def test_native_overlay_coordinates_fonts_and_layering(self) -> None:
    """Overlay positions remain in image pixels, above the background."""

    canvas, _ = self._prepare_canvas(draw=self._draw)
    text, dot_text, time_text = canvas._drawing_elements
    self.assertEqual((text._qt_txt.x(), text._qt_txt.y()), (10, 20))
    self.assertEqual((dot_text._qt_txt.x(), dot_text._qt_txt.y()), (70, 60))
    self.assertEqual((time_text._qt_txt.x(), time_text._qt_txt.y()), (50, 60))
    self.assertEqual(dot_text._qt_dot.rect(),
                     canvas_module.QtCore.QRectF(10, 20, 40, 40))
    self.assertLess(dot_text._qt_dot.zValue(), dot_text._qt_txt.zValue())
    flags = canvas_module.QtWidgets.QGraphicsItem.GraphicsItemFlag
    flag = flags.ItemIgnoresTransformations
    for element, size in zip(canvas._drawing_elements, (10, 16, 38)):
      self.assertIs(element._qt_txt.scene(), canvas.qt_scene)
      self.assertIsNone(element._txt)
      self.assertEqual(element._qt_txt.font().pointSizeF(), size)
      self.assertTrue(element._qt_txt.flags() & flag)

  def test_window_size_is_converted_from_inches_to_screen_pixels(self) -> None:
    """Qt preserves the public inch-based sizing used by the Tk backend."""

    canvas, _ = self._prepare_canvas(window_size=(7, 5))
    screen = canvas._qt_app.primaryScreen()
    dpi_x = screen.logicalDotsPerInchX() if screen is not None else 96
    dpi_y = screen.logicalDotsPerInchY() if screen is not None else 96
    self.assertEqual(canvas._qt_window.width(), round(7 * dpi_x))
    self.assertEqual(canvas._qt_window.height(), round(5 * dpi_y))

  def test_native_dot_colors_clip_and_handle_nan(self) -> None:
    """Coolwarm colors interpolate, saturate, and make NaNs transparent."""

    cases = ((-1, (59, 76, 192, 255)), (0, (59, 76, 192, 255)),
             (0.5, (221, 220, 220, 255)), (1, (180, 4, 38, 255)),
             (2, (180, 4, 38, 255)), (float('nan'), (0, 0, 0, 0)),
             (0.0625, (78, 103, 213, 255)))
    for value, expected in cases:
      with self.subTest(value=value):
        self.assertEqual(Canvas.qt_color(value).getRgb(), expected)
    canvas, source = self._prepare_canvas(draw=self._draw[1:2],
                                          color_range=(1, 5))
    for value, expected in ((-5, cases[0][1]), (10, cases[3][1]),
                             (float('nan'), cases[5][1])):
      with self.subTest(received=value):
        source.send({'pressure': value})
        canvas.loop()
        color = canvas._drawing_elements[0]._qt_dot.brush().color()
        self.assertEqual(color.getRgb(), expected)

  def test_loop_services_idle_gui_and_refits_only_after_resize(self) -> None:
    """Pump events while idle and preserve image aspect ratio on resize."""

    canvas, _ = self._prepare_canvas(draw=self._draw[:1])
    with (patch.object(canvas._qt_app, 'processEvents',
                       wraps=canvas._qt_app.processEvents) as events,
          patch.object(canvas, '_fit_qt_view',
                       wraps=canvas._fit_qt_view) as fit):
      canvas.loop()
      events.assert_called_once_with()
      fit.assert_not_called()
      canvas._qt_window.resize(canvas._qt_window.width() + 100,
                                canvas._qt_window.height() + 50)
      canvas._qt_app.processEvents()
      canvas.loop()
      fit.assert_called_once_with()
      canvas.loop()
      fit.assert_called_once_with()
    self.assertAlmostEqual(canvas._qt_view.transform().m11(),
                           canvas._qt_view.transform().m22())
    self.assertEqual(canvas._qt_view_size,
                     canvas._qt_view.viewport().size())

  def test_missing_and_invalid_images_fail_cleanly(self) -> None:
    """Image-loading failures allow finish to close the partial window."""

    with TemporaryDirectory() as directory:
      missing = Path(directory) / 'missing.png'
      invalid = Path(directory) / 'invalid.png'
      invalid.write_bytes(b'not an image')
      for image, error in ((missing, FileNotFoundError), (invalid, ValueError)):
        with self.subTest(image=image):
          with self.assertRaises(error):
            self._prepare_canvas(image_path=image)
          canvas = self._canvases[-1]
          window = canvas._qt_window
          canvas.finish()
          self.assertIsNone(canvas._qt_window)
          self.assertFalse(window.isVisible())

  def test_closing_window_does_not_stop_the_test(self) -> None:
    """Closing a Canvas does not turn into a global stop request."""

    canvas, _ = self._prepare_canvas(draw=self._draw)
    with patch.object(canvas, 'stop') as stop:
      canvas._qt_window.close()
      canvas.loop()
      stop.assert_not_called()
