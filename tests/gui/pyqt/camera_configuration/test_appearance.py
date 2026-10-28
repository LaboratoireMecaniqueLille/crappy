# coding: utf-8

"""Qt frame and histogram appearance in light and dark palettes."""

from tests.support.gui import require_gui_group

require_gui_group('pyqt')

import numpy as np

from tests.gui.pyqt.camera_configuration._fixtures import PyQtConfigTestCase


class TestAppearance(PyQtConfigTestCase):
  def test_window_renders_in_light_and_dark_palettes(self) -> None:
    """A configuration window remains renderable in light and dark palettes."""

    from PyQt6.QtGui import QColor, QPalette

    config = self.make_config()
    config.show()
    app = config._qt_app
    config._hist = np.full((80, 512), 255, dtype=np.uint8)
    original = app.palette()
    self.addCleanup(app.setPalette, original)
    for background, foreground in ((245, 20), (30, 235)):
      with self.subTest(background=background):
        palette = QPalette(original)
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
          palette.setColor(role, QColor(background, background, background))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
          palette.setColor(role, QColor(foreground, foreground, foreground))
        app.setPalette(palette)
        app.processEvents()
        self.assertEqual(config.palette().color(QPalette.ColorRole.WindowText),
                         QColor(foreground, foreground, foreground))
        self.assertFalse(config.grab().toImage().isNull())
        self.assertTrue(config.isVisible())

  def test_histogram_uses_current_palette(self) -> None:
    """Bars, background, and auto-range markers follow light/dark colors."""

    from PyQt6.QtGui import QColor, QPalette

    config = self.make_config()
    config.show()
    config._qt_app.processEvents()
    hist = np.full((80, 512), 255, dtype=np.uint8)
    hist[:, :150] = 0
    hist[:, 200:300] = 127
    config._hist = hist

    app = config._qt_app
    original = app.palette()
    self.addCleanup(app.setPalette, original)
    palette = QPalette(original)
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 30, 40))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 230, 240))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(60, 170, 180))
    app.setPalette(palette)
    app.processEvents()
    image = config._hist_canvas.pixmap().toImage()
    y = image.height() // 2
    self.assertEqual(image.pixelColor(image.width() // 10, y),
                     QColor(220, 230, 240))
    self.assertEqual(image.pixelColor(image.width() // 2, y),
                     QColor(60, 170, 180))
    self.assertEqual(image.pixelColor(9 * image.width() // 10, y),
                     QColor(20, 30, 40))
