from tests.support.gui import require_gui_group

require_gui_group('pyqt')

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, sentinel
from crappy.blocks.canvas import Canvas
import crappy.blocks.canvas as canvas_module
from tests.support.blocks import BlockTestBase

from tests.gui._support.canvas import CanvasTests


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
    # The scene must still expose its pixmap after temporary wrappers are
    # collected, including on Python versions with different GC timing.
    gc.collect()
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
