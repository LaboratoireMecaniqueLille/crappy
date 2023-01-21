# coding: utf-8

from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, List
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.camera_config import Box, SpotsBoxes
from ...tool.image_processing import DISCorrelTool


class DISCorrelProcess(CameraProcess):
  """"""

  def __init__(self,
               log_queue: Queue,
               log_level: int = 20,
               fields: List[str] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,
               display_freq: bool = False) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     display_freq=display_freq)

    self._discorrel_kw = dict(fields=fields,
                              alpha=alpha,
                              delta=delta,
                              gamma=gamma,
                              finest_scale=finest_scale,
                              init=init,
                              iterations=iterations,
                              gradient_iterations=gradient_iterations,
                              patch_size=patch_size,
                              patch_stride=patch_stride)
    self._residual = residual
    self._discorrel: Optional[DISCorrelTool] = None
    self._img0_set = False

  def set_box(self, box: Box) -> None:
    """"""

    self._discorrel_kw.update(dict(box=box))

  def _init(self) -> None:
    """"""

    self._log(logging.INFO, "Instantiating the Discorrel tool")
    self._discorrel = DISCorrelTool(**self._discorrel_kw)

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return
    self.fps_count += 1

    if not self._img0_set:
      self._log(logging.INFO, "Setting the reference image")
      self._discorrel.set_img0(np.copy(self._img))
      self._img0_set = True
      return

    self._log(logging.DEBUG, "Processing the received image")
    data = self._discorrel.get_data(self._img, self._residual)
    self._send([self._metadata['t(s)'], self._metadata, *data])

    self._send_box(SpotsBoxes(self._discorrel.box))
