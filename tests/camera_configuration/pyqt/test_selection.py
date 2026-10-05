# coding: utf-8

"""Qt box gestures and construction of specialized configuration windows."""

from unittest.mock import patch

from crappy.tool.camera_config import Box, SpotsBoxes
from ._fixtures import PyQtConfigTestCase


class TestSelection(PyQtConfigTestCase):
  def test_standard_configurators_initialize(self) -> None:
    """Each PyQt6 configurator constructs its specialized controls and state."""

    from crappy.tool.camera_config.pyqt import (
      PyQtCameraConfig, PyQtCameraConfigBoxes, PyQtDICVEConfig,
      PyQtDISCorrelConfig, PyQtVideoExtensoConfig)

    roi = Box()
    patches = SpotsBoxes()
    cases = ((PyQtCameraConfig, (), PyQtCameraConfig),
             (PyQtCameraConfigBoxes, (), PyQtCameraConfigBoxes),
             (PyQtDISCorrelConfig, (roi,), PyQtDISCorrelConfig),
             (PyQtDICVEConfig, (patches,), PyQtDICVEConfig),
             (PyQtVideoExtensoConfig,
              (True, None, 10, None, False, False, 5),
              PyQtVideoExtensoConfig))
    for configurator, args, expected in cases:
      with self.subTest(configurator=configurator.__name__):
        config = self.make_config(configurator, *args)
        self.assertIsInstance(config, expected)

    self.assertIs(self.configs[2].get_config()[0], roi)
    self.assertIs(self.configs[3].get_config()[0], patches)
    self.assertIs(self.configs[4].get_config()[0], self.configs[4]._spots)
    self.configs[4]._action_buttons['save_l0'].click()
    self.assertFalse(self.configs[4]._window_closed)
  def test_qt_mouse_selection_and_close_validation(self) -> None:
    """Qt pointer events reach the shared ROI rules and close validation."""

    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from crappy.tool.camera_config.pyqt import camera_config as pyqt_config

    roi = Box()
    from crappy.tool.camera_config.pyqt import PyQtDISCorrelConfig

    config = self.make_config(PyQtDISCorrelConfig, roi)
    config.show()
    config._qt_app.processEvents()
    config._update_img()
    with patch.object(pyqt_config.QMessageBox, 'critical') as dialog:
      config.close()
    dialog.assert_called_once()
    self.assertFalse(config._window_closed)

    geometry = config._display_geometry
    start = QPointF(geometry.left + geometry.image_width * .2,
                    geometry.top + geometry.image_height * .2)
    end = QPointF(geometry.left + geometry.image_width * .7,
                  geometry.top + geometry.image_height * .7)

    def send(kind, point, button, buttons):
      event = QMouseEvent(kind, point, point, button, buttons,
                          Qt.KeyboardModifier.NoModifier)
      config._qt_app.sendEvent(config._img_canvas, event)

    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton,
         Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton,
         Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton,
         Qt.MouseButton.NoButton)

    self.assertFalse(roi.no_points())
    self.assertEqual(roi.sorted(), (20, 70, 20, 70))
    config.close()
    self.assertTrue(config._window_closed)
