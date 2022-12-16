# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from ..tool import GPUCorrel, Spot_boxes
from ..links import Link


class Gpuve_parallel_process(Process):
  """"""

  def __init__(self,
               correls: List[GPUCorrel],
               patches: List[Tuple[int, int, int, int]],
               img0_set: bool) -> None:
    """"""

    super().__init__()

    self._correls = correls
    self._patches = patches

    self._spots = Spot_boxes()
    self._spots.set_spots(patches)

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
            for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
              correl.set_orig(self._img[oy:oy + h,
                                        ox:ox + w].astype(np.float32))
              correl.prepare()
            self._img0_set = True
            continue

          data = [self._metadata['t(s)'], self._metadata]
          for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
            data.extend(correl.get_disp(
              self._img[oy:oy + h, ox:ox + w].astype(np.float32)).tolist())
          self._send(data)

          if self._box_conn is not None:
            self._box_conn.send(self._spots)

    except KeyboardInterrupt:
      pass

    finally:
      for correl in self._correls:
        correl.clean()

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    if isinstance(data, list):
      data = dict(zip(self._labels, data))

    for out in self._outputs:
      out.send(data)
