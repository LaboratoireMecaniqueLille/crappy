# coding: utf-8

"""Cooperative initialization of custom Tk backends and shared base classes."""

from ._fixtures import TkinterConfigTestCase
from crappy.tool.camera_config.base import CameraConfig
from crappy.tool.camera_config.tkinter import (TkinterCameraConfig,
                                                TkinterCameraConfigBoxes)


class TestSubclassing(TkinterConfigTestCase):
  def test_backend_subclass_can_be_constructed(self) -> None:
    """Extending only the Tk backend requires no factory-specific behavior."""

    class CustomTk(TkinterCameraConfig):
      pass

    config = CustomTk(self._camera, self._log_queue, self._log_level,
                       self._freq, None)
    self.addCleanup(config.stop)
    self.assertTrue(config.winfo_exists())

  def test_constructor_extra_argument_contract(self) -> None:
    """The generic window ignores extras, while box-window signatures are strict."""

    config = TkinterCameraConfig(self._camera, self._log_queue, self._log_level,
                                 self._freq, None, 'ignored')
    self.addCleanup(config.stop)
    self.assertTrue(config.winfo_exists())
    with self.assertRaises(TypeError):
      TkinterCameraConfigBoxes(self._camera, self._log_queue, self._log_level,
                                self._freq, None, 'unexpected')

  def test_backend_and_core_subclasses_initialize_cooperatively(self) -> None:
    """A user may extend either layer or combine both in either base order."""

    class ExtendedCore(CameraConfig):
      def __init__(self, camera, log_queue, log_level, max_freq, transform):
        self.core_calls = 1
        super().__init__(camera, log_queue, log_level, max_freq, transform)

      def _create_local_settings(self):
        self.core_hook_called = True
        return super()._create_local_settings()

    class ExtendedTk(TkinterCameraConfig):
      def __init__(self, camera, log_queue, log_level, max_freq, transform):
        self.backend_calls = 1
        super().__init__(camera, log_queue, log_level, max_freq, transform)

    class CoreOnly(TkinterCameraConfig, ExtendedCore):
      pass

    class CoreFirst(ExtendedCore, TkinterCameraConfig):
      pass

    class Both(ExtendedTk, ExtendedCore):
      pass

    for configurator in (ExtendedTk, CoreOnly, CoreFirst, Both):
      with self.subTest(configurator=configurator.__name__):
        window = configurator(self._camera, self._log_queue,
                              self._log_level, self._freq, None)
        self.addCleanup(window.stop)
        self.assertTrue(window.winfo_exists())
        self.assertEqual(hasattr(window, 'backend_calls'),
                         configurator in (ExtendedTk, Both))
        self.assertEqual(hasattr(window, 'core_calls'),
                         configurator is not ExtendedTk)
        if configurator is not ExtendedTk:
          self.assertTrue(window.core_hook_called)
