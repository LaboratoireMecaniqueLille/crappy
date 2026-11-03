# coding: utf-8

"""A minimal concrete configuration for testing shared state without a GUI."""

from collections.abc import Callable

from crappy.tool.camera_config.base import CameraConfig


class RecordingCore(CameraConfig):
  """Headless configuration with no event loop or background resources."""

  def run(self) -> None:
    """No event loop is needed for direct model and interaction checks."""

    pass

  def stop(self) -> None:
    """No GUI or background resources need closing."""

    pass

  def watch_shutdown(self, requested: Callable[[], bool]) -> None:
    """No running event loop needs to poll the shutdown predicate."""

    pass
