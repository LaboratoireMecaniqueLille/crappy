# coding: utf-8

from typing import Any
import numpy as np
from crappy.blocks.camera import Camera

from ..block import BlockTestBase


class CameraWrapperTestBase(BlockTestBase):
  """Common setup for Camera-derived wrapper Block unit tests."""

  @staticmethod
  def camera_kwargs(config: bool = False) -> dict[str, Any]:
    """Returns deterministic Camera arguments without requiring hardware."""

    def image_generator(_, __) -> np.ndarray:
      """Returns a small deterministic greyscale image."""

      return np.zeros((12, 16), dtype=np.uint8)

    return {
      'camera': 'unused',
      'config': config,
      'image_generator': image_generator,
      'img_shape': (12, 16),
      'img_dtype': 'uint8',
    }

  def tearDown(self) -> None:
    """Clears Camera-specific state in addition to common Block state."""

    Camera.cam_count.clear()
    super().tearDown()


class RecordingProcess:
  """Small CameraProcess stand-in recording constructor arguments."""

  instances: list['RecordingProcess'] = list()

  def __init__(self, **kwargs) -> None:
    """Stores constructor arguments for later assertions."""

    self.kwargs = kwargs
    type(self).instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears the instances created by the test double."""

    cls.instances = list()

  @staticmethod
  def is_alive() -> bool:
    """Reports that this stand-in has no running child process."""

    return False
