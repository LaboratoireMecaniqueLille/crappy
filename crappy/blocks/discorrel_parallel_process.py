# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from ..tool import DISCorrel, Box, Spot_boxes
from ..links import Link


class Discorrel_parallel_process(Process):
  """"""

  def __init__(self,
               fields: List[str] = None,
               labels: List[str] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,) -> None:
    """"""

    super().__init__()

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

  def set_box(self, box: Box) -> None:
    """"""

    self._discorrel_kw.update(dict(box=box))

  def run(self) -> None:
    """"""

    try:
      self._discorrel = DISCorrel(**self._discorrel_kw)

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
            self._discorrel.set_img0(np.copy(self._img))
            self._img0_set = True
            continue

          data = self._discorrel.get_data(self._img, self._residual)
          self._send([self._metadata['t(s)'], self._metadata, *data])

          if self._box_conn is not None:
            self._box_conn.send(Spot_boxes(self._discorrel.box))

    except KeyboardInterrupt:
      pass

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    if isinstance(data, list):
      data = dict(zip(self._labels, data))

    for out in self._outputs:
      out.send(data)
