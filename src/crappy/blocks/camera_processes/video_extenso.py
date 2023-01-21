# coding: utf-8

from multiprocessing.queues import Queue
from typing import Optional
import logging
import logging.handlers

from .camera_process import CameraProcess
from ...tool.image_processing import VideoExtensoTool, LostSpotError
from ...tool.camera_config import SpotsDetector


class VideoExtensoProcess(CameraProcess):
  """"""

  def __init__(self,
               detector: SpotsDetector,
               log_queue: Queue,
               log_level: int = 20,
               raise_on_lost_spot: bool = True,
               display_freq: bool = False) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     display_freq=display_freq)

    self._ve: Optional[VideoExtensoTool] = None
    self._detector = detector
    self._raise_on_lost_spot = raise_on_lost_spot
    self._lost_spots = False

  def _init(self) -> None:
    """"""

    self._log(logging.INFO, "Instantiating the VideoExtenso tool")
    self._ve = VideoExtensoTool(spots=self._detector.spots,
                                x_l0=self._detector.x_l0,
                                y_l0=self._detector.y_l0,
                                thresh=self._detector.thresh,
                                log_level=self._log_level,
                                log_queue=self._log_queue,
                                white_spots=self._detector.white_spots,
                                update_thresh=self._detector.update_thresh,
                                num_spots=self._detector.num_spots,
                                safe_mode=self._detector.safe_mode,
                                border=self._detector.border,
                                blur=self._detector.blur)

    self._log(logging.INFO, "Starting the VideoExtenso spot tracker "
                            "processes")
    self._ve.start_tracking()

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return

    if not self._lost_spots:
      self.fps_count += 1
      try:
        self._log(logging.DEBUG, "Processing the received image")
        data = self._ve.get_data(self._img)
        if data is not None:
          self._send([self._metadata['t(s)'], self._metadata, *data])

        self._send_box(self._ve.spots)

      except LostSpotError:
        self._log(logging.INFO, "Spots lost, stopping the spot trackers")
        self._ve.stop_tracking()
        # Raising if specified by the user
        if self._raise_on_lost_spot:
          self._log(logging.ERROR, "Spots lost, stopping the VideoExtenso "
                                   "process")
          raise
        # Otherwise, simply setting a flag so that no additional
        # processing is performed
        else:
          self._lost_spots = True
          self._log(logging.WARNING, "Spots lost, VideoExtenso staying "
                                     "idle until the test ends")

  def _finish(self) -> None:
    """"""

    if self._ve is not None:
      self._log(logging.INFO, "Stopping the spot trackers before returning")
      self._ve.stop_tracking()
