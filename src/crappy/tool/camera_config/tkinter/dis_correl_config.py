# coding: utf-8

"""Tk-backend :class:`~crappy.blocks.DISCorrel` configuration."""

from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config_boxes import TkinterCameraConfigBoxes
from ..base import DISCorrelConfig
from ..config_tools import Box
from ....camera.meta_camera import Camera


class TkinterDISCorrelConfig(DISCorrelConfig, TkinterCameraConfigBoxes):
  """Tkinter window for selecting a :class:`~crappy.blocks.DISCorrel` region of
  interest (ROI).

  A left-button drag replaces any existing correlation
  :class:`~crappy.tool.camera_config.config_tools.Box` when a non-empty
  selection is released. Closing requires a selected ROI. Selection and
  validation rules are inherited from the shared
  :class:`~crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig`.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *DISConfig* to *DISCorrelConfig*
  .. versionchanged:: 2.1.0 renamed from *DISCorrelConfig* to
     *TkinterDISCorrelConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patch: Box) -> None:
    """Builds the window with the provided correlation region.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.

        .. versionadded:: 2.0.0
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.

        .. versionadded:: 2.0.0
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.

        .. versionadded:: 2.0.0
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.

        .. versionadded:: 2.1.0
      patch: :class:`~crappy.tool.camera_config.config_tools.Box` to display
        and update in place. The same
        :class:`~crappy.tool.camera_config.config_tools.Box` is returned by
        :meth:`get_config() <crappy.tool.camera_config.base.dis_correl_config.\
DISCorrelConfig.get_config>` after validation.

        .. versionadded:: 2.0.0
    """

    self._correl_box: Box = patch
    self._draw_correl_box: bool = True

    super().__init__(camera, log_queue, log_level, max_freq, transform)
