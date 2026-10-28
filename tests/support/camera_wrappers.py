# coding: utf-8

from typing import Any
from crappy.blocks.camera import Camera

from tests.support.block_test_case import BlockTestBase
from tests.support.images import zero_frame
from tests.support.resources import preserve_mapping


class CameraWrapperTestBase(BlockTestBase):
  """Common setup for Camera-derived wrapper Block unit tests."""

  @staticmethod
  def camera_kwargs(config: bool = False) -> dict[str, Any]:
    """Returns deterministic Camera arguments without requiring hardware."""

    return {
      'camera': 'unused',
      'config': config,
      'image_generator': zero_frame,
      'img_shape': (12, 16),
      'img_dtype': 'uint8',
    }

  def setUp(self):
    super().setUp()
    preserve_mapping(self, Camera.cam_count)


class RecordingProcess:
  """Small CameraProcess stand-in recording constructor arguments."""

  instances: list['RecordingProcess'] = list()

  def __init__(self, **kwargs) -> None:
    """Stores constructor arguments for later assertions."""

    self.kwargs = kwargs
    self.name = type(self).__name__
    self.closed = False
    type(self).instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears the instances created by the test double."""

    cls.instances = list()

  @staticmethod
  def is_alive() -> bool:
    """Reports that this stand-in has no running child process."""

    return False

  def close(self) -> None:
    """Records the Camera's release of an unstarted process object."""

    self.closed = True
