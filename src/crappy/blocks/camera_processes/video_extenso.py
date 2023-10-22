# coding: utf-8

from multiprocessing.queues import Queue
from typing import Optional
import logging
import logging.handlers
from time import sleep

from .camera_process import CameraProcess
from ...tool.image_processing import VideoExtensoTool, LostSpotError
from ...tool.camera_config import SpotsDetector


class VideoExtensoProcess(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can perform
  video-extensometry by tracking spots on images. It returns the strain and the
  position of the detected spots on the image.

  It is used by the :class:`~crappy.blocks.VideoExtenso` Block to parallelize
  the image processing and the image acquisition. It delegates most of the
  computation to the
  :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`. It is
  from this class that the output values are sent to the downstream Blocks, and
  that the :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` are sent
  to the :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess for
  display.
  """

  def __init__(self,
               detector: SpotsDetector,
               log_queue: Queue,
               log_level: int = 20,
               raise_on_lost_spot: bool = True,
               display_freq: bool = False) -> None:
    """Sets the arguments and initializes the parent class.
    
    Args:
      detector: An instance of the
        :class:`~crappy.tool.camera_config.config_tools.SpotsDetector` Tool,
        containing the coordinates of the detected spots to track. This
        argument is passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`
        and not used in this class.
      log_queue: A :obj:`~multiprocessing.Queue` for sending the log messages
        to the main :obj:`~logging.Logger`, only used in Windows.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
      raise_on_lost_spot: If :obj:`True`, raises an exception when losing the
        spots to track, which stops the test. Otherwise, stops the tracking but
        lets the test go on and silently sleeps.
      display_freq: If :obj:`True`, the looping frequency of this class will be
        displayed while running.
    """

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     display_freq=display_freq)

    self._ve: Optional[VideoExtensoTool] = None
    self._detector = detector
    self._raise_on_lost_spot = raise_on_lost_spot
    self._lost_spots = False

  def init(self) -> None:
    """Instantiates the 
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` and
    starts tracking the spots."""

    self._log(logging.INFO, "Instantiating the VideoExtenso tool")
    self._ve = VideoExtensoTool(spots=self._detector.spots,
                                thresh=self._detector.thresh,
                                log_level=self._log_level,
                                log_queue=self._log_queue,
                                white_spots=self._detector.white_spots,
                                update_thresh=self._detector.update_thresh,
                                safe_mode=self._detector.safe_mode,
                                border=self._detector.border,
                                blur=self._detector.blur)

    self._log(logging.INFO, "Starting the VideoExtenso spot tracker "
                            "processes")
    self._ve.start_tracking()

  def loop(self) -> None:
    """This method grabs the latest frame and gives it for processing to the
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`. Then
    sends the strain and displacement data to the downstream Blocks.

    If there's no new frame grabbed or if the spots were already lost, doesn't
    do anything. When losing the spots, decides whether to raise an exception
    or not based on the user's choice. Also sends the patches for display to
    the :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess.
    """

    # Processing only if the spots haven't been lost
    if not self._lost_spots:
      self.fps_count += 1
      
      # Processing the received frame
      try:
        self._log(logging.DEBUG, "Processing the received image")
        data = self._ve.get_data(self._img)
        
        # Sending the results to the downstream Blocks
        if data is not None:
          self.send([self._metadata['t(s)'], self._metadata, *data])

        # Sending the detected spots to the Displayer for display
        self.send_to_draw(self._ve.spots)

      # In case the spots were just lost
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
    
    # If the spots were lost, avoid spamming the CPU in vain
    else:
      sleep(0.1)

  def finish(self) -> None:
    """Indicates the 
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` to
    stop tracking the spots."""

    if self._ve is not None:
      self._log(logging.INFO, "Stopping the spot trackers before returning")
      self._ve.stop_tracking()
