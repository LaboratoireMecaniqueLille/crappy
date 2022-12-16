# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from ..tool import DISVE, Spot_boxes
from ..links import Link


class Disve_parallel_process(Process):
  """"""

  def __init__(self,
               patches: Spot_boxes,
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

    super().__init__()

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

    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[Tuple[int, int]] = None
    self._box_conn: Optional[Connection] = None
    self._outputs: List[Link] = list()
    self._labels: List[str] = list()

    self._img: Optional[np.ndarray] = None
    self._dtype = None
    self._metadata = {'ImageUniqueID': None}
    self._img0_set = False
    self._lost_patch = False

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
                 event: Event,
                 shape: Tuple[int, int],
                 dtype,
                 box_conn: Optional[Connection],
                 outputs: List[Link],
                 labels: List[str]) -> None:
    """"""

    self._img_array = array
    self._data_dict = data_dict
    self._lock = lock
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype
    self._box_conn = box_conn
    self._outputs = outputs
    self._labels = labels

    self._img = np.empty(shape=shape, dtype=dtype)

  def run(self) -> None:
    """"""

    try:
      self._disve = DISVE(**self._disve_kw)

      while not self._stop_event.is_set():
        process = False
        with self._lock:

          if 'ImageUniqueID' not in self._data_dict:
            continue

          if self._data_dict['ImageUniqueID'] != \
              self._metadata['ImageUniqueID']:
            self._metadata = self._data_dict.copy()
            process = True

            np.copyto(self._img,
                      np.frombuffer(self._img_array.get_obj(),
                                    dtype=self._dtype).reshape(self._shape))

        if process and not self._lost_patch:
          try:

            if not self._img0_set:
              self._disve.set_img0(np.copy(self._img))
              self._img0_set = True
              continue

            data = self._disve.calculate_displacement(self._img)
            self._send([self._metadata['t(s)'], self._metadata, *data])

            if self._box_conn is not None:
              self._box_conn.send(self._disve.patches)

          except RuntimeError:
            if self._raise_on_exit:
              raise
            self._lost_patch = True
            print("[Disve] Patch exiting the ROI, not processing data "
                  "anymore !")

    except KeyboardInterrupt:
      pass

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    if isinstance(data, list):
      data = dict(zip(self._labels, data))

    for out in self._outputs:
      out.send(data)
