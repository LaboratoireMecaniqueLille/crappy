# coding: utf-8

"""Qt application reuse and temporary OpenCV plugin-path isolation."""

import os
import locale
import unittest
from unittest.mock import patch, sentinel

from crappy.tool.camera_config.pyqt import camera_config as pyqt_config


class TestApplication(unittest.TestCase):
  def test_application_creation_preserves_numeric_locale(self) -> None:
    """Qt initialization must not change how Tk parses floating-point values."""

    original = locale.setlocale(locale.LC_NUMERIC)
    self.addCleanup(locale.setlocale, locale.LC_NUMERIC, original)
    for fails in (False, True):
      with self.subTest(fails=fails):
        locale.setlocale(locale.LC_NUMERIC, 'C')

        def create_application(args):
          locale.setlocale(locale.LC_NUMERIC, '')
          if fails:
            raise RuntimeError('application failed')
          return sentinel.application

        with (patch.object(pyqt_config, 'QCoreApplication') as core_application,
              patch.object(pyqt_config, 'QApplication',
                           side_effect=create_application)):
          core_application.instance.return_value = None
          if fails:
            with self.assertRaisesRegex(RuntimeError, 'application failed'):
              pyqt_config.PyQtCameraConfig._get_application()
          else:
            self.assertIs(pyqt_config.PyQtCameraConfig._get_application(),
                           sentinel.application)
          self.assertEqual(locale.setlocale(locale.LC_NUMERIC), 'C')

  def test_opencv_plugin_paths_are_filtered_only_during_creation(self) -> None:
    """Qt creation excludes OpenCV's plugins and restores the original path."""

    opencv = os.path.join('camera', 'cv2', 'qt', 'plugins')
    unrelated = os.path.join('application', 'plugins')
    cases = ((opencv, None),
             (os.pathsep.join((opencv, unrelated)), unrelated),
             (unrelated, unrelated))
    for original, expected in cases:
      with self.subTest(original=original):
        seen = []

        def create_application(args):
          self.assertEqual(args, [])
          seen.append(os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH'))
          return sentinel.application

        with (patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': original}),
              patch.object(pyqt_config, 'QCoreApplication') as core_application,
              patch.object(pyqt_config, 'QApplication') as application):
          core_application.instance.return_value = None
          application.side_effect = create_application
          self.assertIs(pyqt_config.PyQtCameraConfig._get_application(),
                         sentinel.application)
          self.assertEqual(seen, [expected])
          self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], original)

  def test_failed_application_creation_restores_plugin_path(self) -> None:
    """A Qt construction error cannot leave OpenCV's environment modified."""

    original = os.path.join('camera', 'cv2', 'qt', 'plugins')
    with (patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': original}),
          patch.object(pyqt_config, 'QCoreApplication') as core_application,
          patch.object(pyqt_config, 'QApplication') as application):
      core_application.instance.return_value = None
      application.side_effect = RuntimeError('application failed')
      with self.assertRaisesRegex(RuntimeError, 'application failed'):
        pyqt_config.PyQtCameraConfig._get_application()
      self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], original)

  def test_existing_widget_application_is_reused(self) -> None:
    """Reuse the caller's application without touching its plugin paths."""

    class Application:
      pass

    existing = Application()
    original = os.path.join('camera', 'cv2', 'qt', 'plugins')
    with (patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': original}),
          patch.object(pyqt_config, 'QCoreApplication') as core_application,
          patch.object(pyqt_config, 'QApplication', Application)):
      core_application.instance.return_value = existing
      self.assertIs(pyqt_config.PyQtCameraConfig._get_application(), existing)
      self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], original)

  def test_existing_non_widget_application_is_rejected(self) -> None:
    """A QCoreApplication cannot host camera-configuration widgets."""

    class Application:
      pass

    with (patch.object(pyqt_config, 'QCoreApplication') as core_application,
          patch.object(pyqt_config, 'QApplication', Application)):
      core_application.instance.return_value = sentinel.core_application
      with self.assertRaisesRegex(RuntimeError, 'non-widget Qt application'):
        pyqt_config.PyQtCameraConfig._get_application()
