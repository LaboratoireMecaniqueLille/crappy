# coding: utf-8

import logging
import logging.handlers
from time import sleep

from .camera_process import CameraProcess
from ...tool.image_processing import VideoExtensoTool, LostSpotError
from ...tool.camera_config import SpotsBoxes


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

  The initial spots and threshold are injected through :meth:`set_config`
  after the configuration window closes. This process then creates the
  :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` in
  :meth:`init`. The tool, rather than the public Block, owns the creation and
  lifecycle of the individual spot-tracking processes.

  .. versionadded:: 2.0.0
  .. versionchanged:: 2.1.0 receives initial detection results through
     ``set_config()`` and creates runtime helpers only after the process starts
  """

  def __init__(self,
               white_spots: bool,
               num_spots: int | None,
               min_area: int,
               blur: int | None,
               update_thresh: bool,
               safe_mode: bool,
               border: int,
               raise_on_lost_spot: bool) -> None:
    """Sets the arguments and initializes the parent class.
    
    Args:
      white_spots: If :obj:`True`, detects white spots over a black background.
        If :obj:`False`, detects black spots over a white background. Also
        passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.
      num_spots: The number of spots to detect, as an :obj:`int` between `1`
        and `4`. If given, will try to detect exactly that number of spots and
        will fail if not enough spots can be detected. If left to :obj:`None`,
        will detect up to `4` spots, but potentially fewer. This option is not
        used by the runtime process; the public VideoExtenso Block also passes
        it to VideoExtensoConfig for initial detection.
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of the camera
        and the size of the spots to detect. This option is not used by the
        runtime process; VideoExtensoConfig applies it during initial
        detection.
      blur: An :obj:`int`, odd and greater than `1`, defining the size of the
        kernel to use when applying a median blur filter to the image before
        trying to detect spots. Can also be set to :obj:`None`, in which case
        no median blur filter is applied before detecting the spots. Also
        passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.
      update_thresh: If :obj:`True`, the grey level threshold for detecting
        the spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`
        and not used in this class.
      safe_mode: If :obj:`True`, will stop and raise an exception as soon as
        overlapping spots are detected. Otherwise, will first try to reduce the
        detection window to get rid of overlapping. This argument should be
        used when inconsistency in the results may have critical consequences.
        Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`
        and not used in this class.
      border: When searching for the new position of a spot, will search in the
        last known bounding box of this spot plus a few additional pixels in
        each direction. This argument sets the number of additional pixels to
        use. It should be greater than the expected "speed" of the spots, in
        pixels / frame. But if set too high, noise or other spots might hinder
        the detection. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`
        and not used in this class.
      raise_on_lost_spot: If :obj:`True`, raises an exception when losing the
        spots to track, which stops the test. Otherwise, stops the tracking but
        lets the test go on and silently sleeps.
    """

    super().__init__()

    # Video-extensometry options supplied by the public Block
    self._white_spots: bool = white_spots
    self._num_spots: int | None = num_spots
    self._min_area: int = min_area
    self._blur: int | None = blur
    self._update_thresh: bool = update_thresh
    self._safe_mode: bool = safe_mode
    self._border: int = border

    self._ve: VideoExtensoTool | None = None
    self._spots: SpotsBoxes | None = None
    self._thresh: int | None = None
    self._raise_on_lost_spot: bool = raise_on_lost_spot
    self._lost_spots: bool = False

  def init(self) -> None:
    """Instantiates the runtime
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` and
    starts tracking the configured spots.

    The VideoExtensoTool creates and owns one
    :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
    process for each spot.
    """

    if self._log_queue is None:
      raise RuntimeError("At that point the log_queue should be set but it "
                         "isn't")
    if self._spots is None:
      raise RuntimeError("At that point the spots to track should be set but "
                         "they are not")
    if self._thresh is None:
      raise RuntimeError("At that point the threshold should be set but it is "
                         "not")

    self.log(logging.INFO, "Instantiating the VideoExtenso tool")
    self._ve = VideoExtensoTool(spots=self._spots,
                                thresh=self._thresh,
                                log_level=self._log_level,
                                log_queue=self._log_queue,
                                white_spots=self._white_spots,
                                update_thresh=self._update_thresh,
                                safe_mode=self._safe_mode,
                                border=self._border,
                                blur=self._blur)

    self.log(logging.INFO, "Starting the VideoExtenso spot tracker "
                           "processes")
    if self._ve is None:
      raise RuntimeError("At that point the VideoExtensoTool should be set "
                         "but it isn't")
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
      
      # Processing the received frame
      try:
        self.log(logging.DEBUG, "Processing the received image")

        if self._ve is None:
          raise RuntimeError("The VideoExtensoTool isn't initialized")
        if self.img is None:
          raise RuntimeError("The image isn't initialized")
        
        # Sending the results to the downstream Blocks
        if (data := self._ve.get_data(self.img)) is not None:
          self.send([self.metadata['t(s)'], self.metadata, *data])

        # Sending the detected spots to the Displayer for display
        self.send_to_draw(self._ve.spots)

      # In case the spots were just lost
      except LostSpotError:
        if self._ve is None:
          self.log(logging.ERROR, "Trying to stop the spot Trackers but they "
                                  "are not initialized")
        else:
          self.log(logging.INFO, "Spots lost, stopping the spot trackers")
          self._ve.stop_tracking()
        # Raising if specified by the user
        if self._raise_on_lost_spot:
          self.log(logging.ERROR, "Spots lost, stopping the VideoExtenso "
                                  "process")
          raise
        # Otherwise, simply setting a flag so that no additional
        # processing is performed
        else:
          self._lost_spots = True
          self.log(logging.WARNING, "Spots lost, VideoExtenso staying "
                                    "idle until the test ends")
          self.send_to_draw(list())
    
    # If the spots were lost, avoid spamming the CPU in vain
    else:
      self.fps_count -= 1
      sleep(0.1)

  def finish(self) -> None:
    """Indicates the 
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` to
    stop tracking the spots."""

    if self._ve is not None:
      self.log(logging.INFO, "Stopping the spot trackers before returning")
      self._ve.stop_tracking()

  def set_config(self, config: SpotsBoxes, thresh: int) -> None:
    """Stores the initial detection result from
    :class:`~crappy.tool.camera_config.VideoExtensoConfig`.

    Args:
      config: The selected
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` exported by
        :meth:`crappy.tool.camera_config.VideoExtensoConfig.get_config`.
      thresh: The gray-level threshold calculated while detecting those spots.

    These values are received before this process starts and are used by
    :meth:`init` to construct the
    :class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool`.

    .. versionadded:: 2.1.0
    """

    self._spots = config
    self._thresh = thresh
