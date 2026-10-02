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
  """Instantiates an explicit class or one selected from a backend mapping.

  A class provided directly is used as supplied, regardless of
  ``config_backend``. The public Camera Blocks validate ``config_backend``,
  subclasses select their configurator through a class attribute.

  Args:
    configurator: Concrete CameraConfig subclass, or a mapping from
      backend names to those classes.
    camera: Camera to configure.
    config_backend: Backend name used to select a class from *configurator*
      when it is a mapping.
    log_queue: Queue used to send configuration log messages.
    log_level: Logging level for the configuration window.
    max_freq: Maximum preview acquisition frequency.
    transform: Optional transformation applied to acquired images.
    *args: Additional positional arguments passed to the selected class.
    **kwargs: Additional keyword arguments passed to the selected class.

  Returns:
    The instantiated configuration window.
  """

  selected = (configurator[config_backend]
              if isinstance(configurator, Mapping) else configurator)
  return selected(camera, log_queue, log_level, max_freq, transform,
                  *args, **kwargs)
