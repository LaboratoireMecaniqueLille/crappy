# coding: utf-8

"""Select and instantiate the requested camera configuration GUI backend."""

from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping
from importlib import import_module
from multiprocessing.queues import Queue
from typing import Any, TYPE_CHECKING

from ... import _validation as check

if TYPE_CHECKING:
  import numpy as np
  from .base import CameraConfig
  from ...camera.meta_camera import Camera
  from ..._failure import FailureManager


class LazyConfiguratorMapping(Mapping[str, type['CameraConfig']]):
  """Read-only backend choices that import camera configuration classes.

  .. versionadded:: 2.2.0
  """

  def __init__(self, references: Mapping[str, tuple[str, str]]) -> None:
    """Stores the import references without loading their implementations.

    Args:
      references: Backend names mapped to ``(module_name, class_name)`` pairs.
        Relative module names are resolved from ``crappy.tool.camera_config``.
    """

    self._references: dict[str, tuple[str, str]] = dict(references)
    self._classes: dict[str, type[CameraConfig]] = dict()

  def __getitem__(self, backend: str) -> type[CameraConfig]:
    """Returns the cached class, resolving it on first access."""

    if backend not in self._classes:
      module, name = self._references[backend]
      self._classes[backend] = getattr(import_module(module, __package__),
                                       name)

    return self._classes[backend]

  def __iter__(self) -> Iterator[str]:
    """Iterates over backend names without importing classes."""

    return iter(self._references)

  def __len__(self) -> int:
    """Returns the number of declared backends without importing classes."""

    return len(self._references)

  def __contains__(self, backend: object) -> bool:
    """Checks whether a backend is declared without importing its class."""

    return backend in self._references


def resolve_configurator(configurator: type[CameraConfig] |
                         Mapping[str, type[CameraConfig]],
                         config_backend: str) -> type[CameraConfig]:
  """Returns the selected configurator class.

  A class supplied directly is returned unchanged.

  .. versionadded:: 2.2.0
  """

  check.string(config_backend, 'config_backend')
  if not isinstance(configurator, Mapping) and not callable(configurator):
    raise TypeError("configurator must be callable or a mapping of backends "
                    "to callables")
  selected = (configurator[config_backend] if isinstance(configurator, Mapping)
              else configurator)
  check.callable_(selected, 'selected configurator')

  return selected


def create_configurator(configurator: type[CameraConfig] |
                        Mapping[str, type[CameraConfig]],
                        camera: Camera,
                        config_backend: str,
                        log_queue: Queue,
                        log_level: int | None,
                        max_freq: float | None,
                        transform: Callable[[np.ndarray], np.ndarray] | None,
                        failure_manager: FailureManager,
                        *args: Any,
                        **kwargs: Any) -> CameraConfig:
  """Creates a configuration window for the selected backend.

  A class supplied directly is used regardless of ``config_backend``. A
  mapping selects the class for that backend. The returned window is ready
  for a call to its run method.

  Built-in mappings load and cache the selected real class on first indexing.
  Listing their keys does not import GUI implementations. User-supplied classes
  and mappings retain the same construction and forwarding behavior.

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
    failure_manager: The :class:`~crappy._failure.FailureManager` to use for
      reporting caught exceptions.

      .. versionadded:: 2.2.0
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

  selected = resolve_configurator(configurator, config_backend)
  return selected(camera, log_queue, log_level, max_freq, transform,
                  failure_manager, *args, **kwargs)
