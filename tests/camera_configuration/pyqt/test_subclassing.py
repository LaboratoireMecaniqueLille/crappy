# coding: utf-8

"""Cooperative initialization of custom Qt backends and shared base classes."""

from crappy.tool.camera_config import SpotsBoxes
from crappy.tool.camera_config.base import CameraConfig, DICVEConfig
from crappy.tool.camera_config.pyqt import PyQtCameraConfig, PyQtCameraConfigBoxes
from ._fixtures import PyQtConfigTestCase


class TestSubclassing(PyQtConfigTestCase):
  def test_backend_subclass_can_be_constructed(self) -> None:
    """Extending only the Qt backend requires no factory-specific behavior."""

    class CustomQt(PyQtCameraConfig):
      pass

    self.assertIsInstance(self.make_config(CustomQt), CustomQt)

  def test_constructor_extra_argument_contract(self) -> None:
    """The generic window ignores extras, while box-window signatures are strict."""

    self.assertIsInstance(self.make_config(PyQtCameraConfig, 'ignored'),
                          PyQtCameraConfig)
    with self.assertRaises(TypeError):
      self.make_config(PyQtCameraConfigBoxes, 'unexpected')

  def test_specialized_backend_and_core_subclasses_compose(self) -> None:
    """A shared core subclass can wrap a specialized Qt window via MRO."""

    from crappy.tool.camera_config.pyqt import PyQtDICVEConfig

    class ExtendedCore(CameraConfig):
      def __init__(self, camera, log_queue, log_level, max_freq, transform,
                   *args, **kwargs):
        self.core_calls = 1
        super().__init__(camera, log_queue, log_level, max_freq, transform,
                         *args, **kwargs)

      def _is_on_image(self, x, y):
        self.core_hook_called = True
        return super()._is_on_image(x, y)

    class ExtendedQt(PyQtDICVEConfig):
      def __init__(self, *args, **kwargs):
        self.backend_calls = 1
        super().__init__(*args, **kwargs)

    class Combined(ExtendedQt, ExtendedCore):
      pass

    class CoreFirst(ExtendedCore, ExtendedQt):
      pass

    for configurator in (Combined, CoreFirst):
      with self.subTest(configurator=configurator.__name__):
        config = self.make_config(configurator, SpotsBoxes())
        self.assertEqual(config.core_calls, 1)
        self.assertEqual(config.backend_calls, 1)
        config._is_on_image(0, 0)
        self.assertTrue(config.core_hook_called)
        self.assertIsNotNone(config._patch_size)
        self.assertIsNotNone(config._qt_app)

    class ExtendedConfig(DICVEConfig):
      def _create_local_settings(self):
        self.behavior_hook_called = True
        return super()._create_local_settings()

    class ConfigQt(ExtendedConfig, PyQtDICVEConfig):
      pass

    config = self.make_config(ConfigQt, SpotsBoxes())
    self.assertTrue(config.behavior_hook_called)
    self.assertIsNotNone(config._patch_size)
