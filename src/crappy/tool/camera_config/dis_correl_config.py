# coding: utf-8

"""Tk-backed DISCorrel configuration."""

from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box
from .selection_behavior import DISCorrelBehavior
from ...camera.meta_camera import Camera


class DISCorrelConfig(DISCorrelBehavior, CameraConfigBoxes):
  """Configure the image region used by a DISCorrel Block.

  Draw a box with the left mouse button to replace the correlation ROI. The
  supplied :class:`~crappy.tool.camera_config.config_tools.Box` is updated in
  place when a valid selection is released. Selection and validation rules are
  shared by :class:`DISCorrelBehavior`, Tk only supplies the interface.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *DISConfig* to *DISCorrelConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patch: Box) -> None:
    """Initialize the configurator with the caller-owned correlation ROI.

    Args:
      camera: The :class:`~crappy.camera.meta_camera.camera.Camera` object in
        charge of acquiring the images.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      max_freq: The maximum frequency this window is allowed to loop at. It is
        simply the ``freq`` attribute of the :class:`~crappy.blocks.Camera`
        Block.

        .. versionadded:: 2.0.0
      transform: A callable taking an image as an argument, and returning a
        transformed image as an output.

        .. versionadded:: 2.1.0
      patch: The :class:`~crappy.tool.camera_config.config_tools.Box` container
        that will save the information on the patch where to perform image
        correlation.

        .. versionadded:: 2.0.0
    """

    self._correl_box: Box = patch
    self._draw_correl_box: bool = True

    super().__init__(camera, log_queue, log_level, max_freq, transform)
