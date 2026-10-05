# coding: utf-8

"""Tk-backend :class:`~crappy.blocks.DICVE` configuration."""

from collections.abc import Callable
from multiprocessing.queues import Queue
import numpy as np

from .camera_config_boxes import TkinterCameraConfigBoxes
from ..base import DICVEConfig
from ..config_tools import SpotsBoxes
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import CameraScaleSetting


class TkinterDICVEConfig(DICVEConfig, TkinterCameraConfigBoxes):
  """Tkinter window for selecting digital image correlation tracking patches.

  Dragging a rectangle large enough for the Patch size setting positions four
  patches around its edges. Apply Settings or Auto apply controls the local
  patch size. If any, the supplied
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` collection is
  updated initially. A valid close saves initial horizontal and vertical patch
  distances. Selection rules are inherited from the shared
  :class:`~crappy.tool.camera_config.base.dic_ve_config.DICVEConfig`.

  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *DISVE_config* to *DICVEConfig*
  .. versionchanged:: 2.1.0 renamed from *DICVEConfig* to *TkinterDICVEConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               patches: SpotsBoxes) -> None:
    """Builds the window with the provided tracking-patch collection.

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

      patches: :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes`
        collection to display and update in place. The same collection is
        returned by
        :meth:`get_config() <crappy.tool.camera_config.base.dic_ve_config.\
DICVEConfig.get_config>` after validation.
    """

    self._patch_size: CameraScaleSetting | None = None

    super().__init__(camera, log_queue, log_level, max_freq, transform)

    self._spots: SpotsBoxes = patches
