# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from ..tool import GPUCorrel
from ..links import Link


class Gpucorrel_parallel_process(Process):
  """"""

  def __init__(self,
               correl: GPUCorrel,
               img0_set: bool,
               discard_limit: float = 3,
               discard_ref: int = 5,
               calc_res: bool = False) -> None:
    """"""

    super().__init__()

    self._correl = correl

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
    self._img0_set = img0_set

    self._res_history = [np.inf]
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._calc_res = calc_res

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

        if process:

          if not self._img0_set:
            self._correl.set_img_size(self._img.shape)
            self._correl.set_orig(self._img.astype(np.float32))
            self._correl.prepare()
            self._img0_set = True
            continue

          data = [self._metadata['t(s)'], self._metadata]
          data += self._correl.get_disp(self._img.astype(np.float32)).tolist()

          if self._calc_res:
            res = self._correl.get_res()
            data.append(res)

            if self._discard_limit:
              self._res_hist.append(res)
              self._res_hist = self._res_hist[-self._discard_ref - 1:]

              if res > self._discard_limit * np.average(self._res_hist[:-1]):
                print("[GPUCorrel] Residual too high, not sending values")
                continue

          self._send(data)

    except KeyboardInterrupt:
      pass

    finally:
      self._correl.clean()

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    if isinstance(data, list):
      data = dict(zip(self._labels, data))

    for out in self._outputs:
      out.send(data)
