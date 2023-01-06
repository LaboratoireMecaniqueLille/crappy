# coding: utf-8

from multiprocessing.queues import Queue
from typing import Optional
import numpy as np
import logging
import logging.handlers

from .camera_process import Camera_process
from ..tool import DISVE, Spot_boxes


class Disve_parallel_process(Camera_process):
  """"""

  def __init__(self,
               patches: Spot_boxes,
               log_queue: Queue,
               log_level: int = 20,
               method: str = 'Disflow',
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               border: float = 0.2,
               safe: bool = True,
               follow: bool = True,
               raise_on_exit: bool = True) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level)

    self._disve_kw = dict(patches=patches,
                          method=method,
                          alpha=alpha,
                          delta=delta,
                          gamma=gamma,
                          finest_scale=finest_scale,
                          iterations=iterations,
                          gradient_iterations=gradient_iterations,
                          patch_size=patch_size,
                          patch_stride=patch_stride,
                          border=border,
                          safe=safe,
                          follow=follow)
    self._raise_on_exit = raise_on_exit
    self._disve: Optional[DISVE] = None
    self._img0_set = False
    self._lost_patch = False

  def _init(self) -> None:
    """"""

    self._log(logging.INFO, "Instantiating the Disve tool")
    self._disve = DISVE(**self._disve_kw)

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return

    if not self._lost_patch:
      try:

        if not self._img0_set:
          self._log(logging.INFO, "Setting the reference image")
          self._disve.set_img0(np.copy(self._img))
          self._img0_set = True
          return

        self._log(logging.DEBUG, "Processing the received image")
        data = self._disve.calculate_displacement(self._img)
        self._send([self._metadata['t(s)'], self._metadata, *data])

        self._send_box(self._disve.patches)

      except RuntimeError as exc:
        if self._raise_on_exit:
          self._logger.exception("Patch exiting the ROI !", exc_info=exc)
          raise
        self._lost_patch = True
        self._log(logging.WARNING, "Patch exiting the ROI, not processing "
                                   "data anymore !")
