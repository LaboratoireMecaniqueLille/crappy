# coding: utf-8

"""The shared classes and Tk backend must remain usable without PyQt6."""

import subprocess
import sys
import textwrap
import unittest


class TestOptionalImports(unittest.TestCase):
  def test_configuration_classes_import_without_pyqt(self) -> None:
    """A fresh interpreter cannot inherit already-imported Qt modules."""

    script = textwrap.dedent('''
        import builtins
        from inspect import isabstract

        original_import = builtins.__import__

        def without_pyqt(name, *args, **kwargs):
          if name == 'PyQt6' or name.startswith('PyQt6.'):
            raise ModuleNotFoundError('PyQt6 intentionally unavailable')
          return original_import(name, *args, **kwargs)

        builtins.__import__ = without_pyqt

        from crappy.camera.meta_camera import Camera
        from crappy.tool.camera_config.base import CameraConfig
        from crappy.tool.camera_config.tkinter import (
          TkinterCameraConfig, TkinterCameraConfigBoxes, TkinterDICVEConfig,
          TkinterDISCorrelConfig, TkinterVideoExtensoConfig)
        from crappy.tool.camera_config.pyqt import (
          PyQtCameraConfig, PyQtCameraConfigBoxes, PyQtDICVEConfig,
          PyQtDISCorrelConfig, PyQtVideoExtensoConfig)

        classes = (TkinterCameraConfig, TkinterCameraConfigBoxes,
                   TkinterDICVEConfig, TkinterDISCorrelConfig,
                   TkinterVideoExtensoConfig, PyQtCameraConfig,
                   PyQtCameraConfigBoxes, PyQtDICVEConfig,
                   PyQtDISCorrelConfig, PyQtVideoExtensoConfig)
        for cls in classes:
          assert issubclass(cls, CameraConfig), cls
          assert not isabstract(cls), cls
          assert cls.__mro__.count(CameraConfig) == 1, cls

        class NoHardwareCamera(Camera):
          def get_image(self):
            return None

        try:
          PyQtCameraConfig(NoHardwareCamera(), None, None, None, None)
        except RuntimeError as error:
          assert 'PyQt6' in str(error), error
        else:
          raise AssertionError('Missing PyQt6 should be reported on use')
    ''')
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
