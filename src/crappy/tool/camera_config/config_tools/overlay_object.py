# coding: utf-8

from numpy import ndarray
import logging
from multiprocessing import current_process
from typing import Optional


class Overlay:
  """This class is the base class for all the classes adding overlays on top of
  the images displayed by a :class:`~crappy.blocks.camera_processes.Displayer`
  Process of a :class:`~crappy.blocks.Camera` Block.

  Also used for drawing overlays on top of the images in the
  :class:`~crappy.tool.camera_config.CameraConfig` window, for the children of
  the Camera Block supporting it.

  It is mainly useful for providing the :meth:`log` method, and creating a
  clear architecture. It is also relevant to use for type-hinting.

  .. versionadded:: 2.0.0
  """

  def __init__(self) -> None:
    """Simply initializes the logger to :obj:`None`."""

    self._logger: Optional[logging.Logger] = None

  def draw(self, img: ndarray) -> None:
    """This method takes the image to display as an input, draws an overlay on
    top of it, and returns the modified image.

    It is meant to be overriden by subclasses of this class.
    """

    ...

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the Overlay class.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(f"{current_process().name}."
                                       f"{type(self).__name__}")

    self._logger.log(log_level, msg)
