# coding: utf-8

"""Select and instantiate the requested camera configuration GUI backend."""

from collections.abc import Callable, Mapping
from multiprocessing.queues import Queue
from typing import Any
import numpy as np

from .base import CameraConfig
from ...camera.meta_camera import Camera


def create_configurator(configurator: type[CameraConfig] |
                        Mapping[str, type[CameraConfig]],
                        camera: Camera,
                        config_backend: str,
                        log_queue: Queue,
                        log_level: int | None,
                        max_freq: float | None,
                        transform: Callable[[np.ndarray], np.ndarray] | None,
                        *args: Any,
                        **kwargs: Any) -> CameraConfig:
  """Instantiates a configuration class from an explicit given type or a
  backend mapping.

  A class supplied directly is used regardless of ``config_backend``. Public
  :class:`Camera Blocks <crappy.blocks.Camera>` validate backend selection,
  while :class:`~crappy.blocks.meta_block.block.Block` subclasses choose
  configurator classes through their configurator attribute. This helper does
  not register classes, validate arguments, or start the configuration window.

  Args:
    configurator:
      :class:`~crappy.tool.camera_config.base.camera_config.CameraConfig`
      subclass or mapping from backend names to classes.
    camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
      providing preview images and adjustable settings.
    config_backend: Mapping key selecting the requested backend. Ignored when
      configurator is a class.
    log_queue: Crappy logging queue, forwarded to the histogram worker.
    log_level: Script logging level, or :obj:`None` to disable worker logging.
      The window uses the logger configured by its owning
      :class:`~crappy.blocks.meta_block.block.Block`.
    max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
      this limit, but acquisition and rendering may reduce the achieved rate.
    transform: :obj:`~collections.abc.Callable` applied to acquired images
      before preview conversion and image-format reporting, or :obj:`None` to
      leave them unchanged.
    *args: Additional positional arguments passed to the selected class.
    **kwargs: Additional keyword arguments passed to the selected class.

  Returns:
    Initialized configurator. Call
    :meth:`run() <crappy.tool.camera_config.base.camera_config.CameraConfig.\
run>` to start configuration.

  Raises:
    KeyError: If a mapping does not contain config_backend.

  .. versionadded:: 2.1.0
  """

  selected = (configurator[config_backend]
              if isinstance(configurator, Mapping) else configurator)
  return selected(camera, log_queue, log_level, max_freq, transform,
                  *args, **kwargs)
