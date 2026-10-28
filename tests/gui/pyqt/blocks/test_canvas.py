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
    """Qt can display its image and overlays without accessing another toolkit."""

    with (patch.object(canvas_module, 'mpl', sentinel.unused_mpl),
          patch.object(canvas_module, 'mpl_figure', sentinel.unused_figure),
          patch.object(canvas_module, 'mpl_image', sentinel.unused_image),
          patch.object(canvas_module, 'mpl_patches', sentinel.unused_patches),
          patch.object(canvas_module, 'backend_tkagg', sentinel.unused_tkagg),
          patch.object(canvas_module, 'tk', sentinel.unused_tk)):
      canvas, source = self._prepare_canvas(draw=self._draw, color_range=(1, 5),
                                            title='<b>Test Canvas</b>')
      source.send({'temperature': 3.5, 'pressure': 4.0})
      canvas.loop()

    widgets = canvas_module.QtWidgets
    self.assertIsInstance(canvas._qt_window, widgets.QWidget)
    self.assertTrue(canvas._qt_window.isVisible())
    self.assertIs(canvas._qt_view.scene(), canvas.qt_scene)
    self.assertEqual(canvas._qt_window.windowTitle(), '<b>Test Canvas</b>')
    labels = canvas._qt_window.findChildren(widgets.QLabel)
    title = next(label for label in labels if label.text() == '<b>Test Canvas</b>')
    self.assertEqual(title.textFormat(),
                     canvas_module.QtCore.Qt.TextFormat.PlainText)
    numeric_labels = []
    for label in labels:
      try:
        numeric_labels.append(float(label.text()))
      except ValueError:
        pass
    self.assertIn(1, numeric_labels)
    self.assertIn(5, numeric_labels)
    self.assertTrue(any(not label.pixmap().isNull() for label in labels))

    gc.collect()
    image = canvas_module.QtGui.QImage(str(self._image))
    self.assertTrue(any(isinstance(item, widgets.QGraphicsPixmapItem)
                        and item.pixmap().size() == image.size()
                        for item in canvas.qt_scene.items()))
    self.assertTrue(canvas.qt_scene.sceneRect().contains(
        canvas_module.QtCore.QRectF(image.rect())))

  def test_native_overlay_positions_and_stacking(self) -> None:
    """Requested pixel positions and label stacking survive native rendering."""

    canvas, _ = self._prepare_canvas(draw=self._draw)
    text, dot_text, time_text = canvas._drawing_elements
    self.assertEqual((text._qt_txt.x(), text._qt_txt.y()), (10, 20))
    self.assertEqual((time_text._qt_txt.x(), time_text._qt_txt.y()), (50, 60))
    self.assertEqual(dot_text._qt_dot.rect().center(),
                     canvas_module.QtCore.QPointF(30, 40))
    stacking = canvas.qt_scene.items()
    self.assertLess(stacking.index(dot_text._qt_txt), stacking.index(dot_text._qt_dot))
    for element in canvas._drawing_elements:
      self.assertIs(element._qt_txt.scene(), canvas.qt_scene)

  def test_window_size_is_converted_from_inches_to_screen_pixels(self) -> None:
    """Qt preserves the public inch-based sizing used by the Tk backend."""

    canvas, _ = self._prepare_canvas(window_size=(7, 5))
    screen = canvas._qt_app.primaryScreen()
    dpi_x = screen.logicalDotsPerInchX() if screen is not None else 96
    dpi_y = screen.logicalDotsPerInchY() if screen is not None else 96
    self.assertAlmostEqual(canvas._qt_window.width(), 7 * dpi_x, delta=2)
    self.assertAlmostEqual(canvas._qt_window.height(), 5 * dpi_y, delta=2)

  def test_native_dot_colors_clip_and_handle_nan(self) -> None:
    """Colors distinguish values, clip at the range ends, and hide missing data."""

    low, high = Canvas.qt_color(0), Canvas.qt_color(1)
    self.assertNotEqual(low, high)
    self.assertEqual(Canvas.qt_color(-1), low)
    self.assertEqual(Canvas.qt_color(2), high)
    for fraction in (0.0625, 0.5):
      color = Canvas.qt_color(fraction)
      self.assertNotIn(color, (low, high))
      self.assertGreater(color.alpha(), 0)
    self.assertEqual(Canvas.qt_color(float('nan')).alpha(), 0)

    canvas, source = self._prepare_canvas(draw=self._draw[1:2], color_range=(1, 5))
    for value, expected in ((-5, low), (10, high), (float('nan'), None)):
      with self.subTest(received=value):
        source.send({'pressure': value})
        canvas.loop()
        color = canvas._drawing_elements[0]._qt_dot.brush().color()
        if expected is None:
          self.assertEqual(color.alpha(), 0)
        else:
          self.assertEqual(color, expected)

  def test_loop_services_idle_gui_and_preserves_aspect_on_resize(self) -> None:
    """Idle UI work runs and resizing preserves the background's aspect ratio."""

    from PyQt6.QtCore import QTimer

    canvas, _ = self._prepare_canvas(draw=self._draw[:1])
    handled = []
    QTimer.singleShot(0, lambda: handled.append(True))
    canvas.loop()
    self.assertEqual(handled, [True])
    initial_scale = canvas._qt_view.transform().m11()
    canvas._qt_window.resize(canvas._qt_window.width() + 100,
                              canvas._qt_window.height() + 50)
    canvas._qt_app.processEvents()
    canvas.loop()
    transform = canvas._qt_view.transform()
    self.assertNotEqual(transform.m11(), initial_scale)
    self.assertAlmostEqual(transform.m11(), transform.m22())

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
