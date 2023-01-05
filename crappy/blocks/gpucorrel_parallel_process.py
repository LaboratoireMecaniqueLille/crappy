# coding: utf-8

from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, List, Union
from pathlib import Path
import logging
import logging.handlers

from .camera_process import Camera_process
from ..tool import GPUCorrel


class Gpucorrel_parallel_process(Camera_process):
  """"""

  def __init__(self,
               log_queue: Queue,
               parent_name: str,
               log_level: int = 20,
               discard_limit: float = 3,
               discard_ref: int = 5,
               calc_res: bool = False,
               img_ref: Optional[np.ndarray] = None,
               verbose: int = 0,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               fields: Optional[List[str]] = None,
               mask: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     parent_name=parent_name,
                     log_level=log_level)

    self._gpucorrel_kw = dict(context=None,
                              verbose=verbose,
                              levels=levels,
                              resampling_factor=resampling_factor,
                              kernel_file=kernel_file,
                              iterations=iterations,
                              fields=fields,
                              ref_img=img_ref,
                              mask=mask,
                              mul=mul)

    self._correl: Optional[GPUCorrel] = None
    self._img_ref = img_ref
    self._img0_set = img_ref is not None

    self._res_history = [np.inf]
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._calc_res = calc_res

  def _init(self) -> None:
    """"""

    self._log(logging.INFO, "Instantiating the GPUCorrel tool")
    self._gpucorrel_kw.update(
      logger_name=f'crappy.{self._parent_name}.Process')
    self._correl = GPUCorrel(**self._gpucorrel_kw)

    if self._img_ref is not None:
      self._log(logging.INFO, "Initializing the GPUCorrel tool with the "
                              "given reference image")
      self._correl.set_img_size(self._img_ref.shape)
      self._correl.set_orig(self._img_ref.astype(np.float32))
      self._log(logging.INFO, "Preparing the GPUCorrel tool")
      self._correl.prepare()

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return

    if not self._img0_set:
      self._log(logging.INFO, "Setting the reference image")
      self._correl.set_img_size(self._img.shape)
      self._correl.set_orig(self._img.astype(np.float32))
      self._correl.prepare()
      self._img0_set = True
      return

    self._log(logging.DEBUG, "Processing the received image")
    data = [self._metadata['t(s)'], self._metadata]
    data += self._correl.get_disp(self._img.astype(np.float32)).tolist()

    if self._calc_res:
      self._log(logging.DEBUG, "Calculating the residuals")
      res = self._correl.get_res()
      data.append(res)

      if self._discard_limit:
        self._log(logging.DEBUG, "Adding residuals to the residuals "
                                 "history")
        self._res_history.append(res)
        self._res_history = self._res_history[-self._discard_ref - 1:]

        if res > self._discard_limit * np.average(self._res_history[:-1]):
          self._log(logging.WARNING, "Residual too high, not sending "
                                     "values")
          return

    self._send(data)

  def _finish(self) -> None:
    """"""

    if self._correl is not None:
      self._log(logging.INFO, "Cleaning up the GPUCorrel tool")
      self._correl.clean()
