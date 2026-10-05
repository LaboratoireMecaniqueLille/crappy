# coding: utf-8

"""Qt wheel and right-button events mapped to shared zoom and pan behavior."""

from ._fixtures import PyQtConfigTestCase


class TestInteraction(PyQtConfigTestCase):
  def test_qt_wheel_and_right_drag_use_shared_zoom_and_pan(self) -> None:
    """Qt wheel and right-button gestures operate on display coordinates."""

    from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent, QWheelEvent

    config = self.make_config()
    config.show()
    config._qt_app.processEvents()
    config._update_img()
    center = QPointF(config._display_geometry.width / 2,
                     config._display_geometry.height / 2)
    wheel = QWheelEvent(center, center, QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    config._qt_app.sendEvent(config._img_canvas, wheel)
    self.assertGreater(config._display_state.zoom_percent, 100)

    press = QMouseEvent(QEvent.Type.MouseButtonPress, center, center,
                        Qt.MouseButton.RightButton,
                        Qt.MouseButton.RightButton,
                        Qt.KeyboardModifier.NoModifier)
    config._qt_app.sendEvent(config._img_canvas, press)
    before = config._zoom_values.x_low
    moved = QPointF(center.x() + 20, center.y())
    drag = QMouseEvent(QEvent.Type.MouseMove, moved, moved,
                       Qt.MouseButton.NoButton,
                       Qt.MouseButton.RightButton,
                       Qt.KeyboardModifier.NoModifier)
    config._qt_app.sendEvent(config._img_canvas, drag)
    self.assertLess(config._zoom_values.x_low, before)
