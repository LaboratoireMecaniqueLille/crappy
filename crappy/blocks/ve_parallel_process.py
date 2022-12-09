# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from ..tool.videoextenso import VideoExtenso, LostSpotError
from ..links import Link


class Ve_parallel_process(Process):
  """"""

  def __init__(self,
               ve: VideoExtenso,
               raise_on_lost_spot: bool = True) -> None:
    """"""

    super().__init__()

    self._ve = ve
    self._raise_on_lost_spot = raise_on_lost_spot

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
    self._lost_spots = False

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
      self._ve.start_tracking()

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

        if process and not self._lost_spots:
          try:
            data = self._ve.get_data(self._img)
            if data is not None:
              self._send([self._metadata['t(s)'], self._metadata, *data])

            if self._box_conn is not None:
              self._box_conn.send(self._ve.spots)

          except LostSpotError:
            self._ve.stop_tracking()
            # Raising if specified by the user
            if self._raise_on_lost_spot:
              raise
            # Otherwise, simply setting a flag so that no additional
            # processing is performed
            else:
              self._lost_spots = True
              print("[VideoExtenso] Spots lost, not processing data anymore !")

    except KeyboardInterrupt:
      pass

    finally:
      self._ve.stop_tracking()

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    if isinstance(data, list):
      data = dict(zip(self._labels, data))

    for out in self._outputs:
      out.send(data)
