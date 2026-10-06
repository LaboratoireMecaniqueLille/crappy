# coding: utf-8

from numpy import ndarray
import logging
from multiprocessing import current_process
from abc import ABC, abstractmethod


class Overlay(ABC):
  """Abstract overlay that draws into a display image in place.

  Used by the all-in-one image
  :class:`~crappy.blocks.camera_processes.Displayer` and the
  :class:`~crappy.blocks.vision.block.VisionBlock`
  :class:`~crappy.blocks.vision.ImageDisplayer`. Subclasses implement
  :meth:`draw() <crappy.tool.camera_config.config_tools.Overlay.draw>` and can
  use the process-local
  :meth:`log() <crappy.tool.camera_config.base.camera_config.CameraConfig.log>`
  helper. Configuration selection classes also use
  :class:`~crappy.tool.camera_config.config_tools.Box` containers, with drawing
  adapted to their preview geometry.

  .. versionadded:: 2.0.0
  """

  def __init__(self) -> None:
    """Simply initializes the logger to :obj:`None`."""

    super().__init__()

    self._logger: logging.Logger | None = None

  @abstractmethod
  def draw(self, img: ndarray) -> None:
    """Draws an overlay into the supplied image in place.

    Args:
      img: Display image array to modify.
    """

    ...

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the
    :class:`~crappy.tool.camera_config.config_tools.Overlay` class.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(f"{current_process().name}."
                                       f"{type(self).__name__}")

    self._logger.log(log_level, msg)
