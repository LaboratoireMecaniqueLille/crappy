# coding: utf-8

from collections.abc import Sequence
import logging

from .block import VisionBlock, ConfigRequest
from ...tool.camera_config import VideoExtensoConfig
from ...tool.image_processing import VideoExtensoTool, LostSpotError
from ...tool.camera_config.config_tools import SpotsBoxes


class VideoExtensoProcessor(VisionBlock):
  """Performs video-extensometry by tracking spots in a stream of images.

  Unlike :class:`~crappy.blocks.VideoExtenso`, this Block does not acquire,
  display, or record images itself. It only implements the image-processing
  stage and must receive images from exactly one upstream
  :class:`~crappy.links.ImageLink`. Image acquisition is normally handled by a
  :class:`~crappy.blocks.vision.CameraSource`. Displaying or recording the same
  images can be achieved by connecting additional VisionBlocks to that source.

  Before the test starts, the VideoExtensoProcessor asks its upstream image
  source to run a :class:`~crappy.tool.camera_config.VideoExtensoConfig`
  window. This window lets the user adjust the Camera settings, visualize the
  acquired images, and detect or manually select the spots to track. When the
  source is a :class:`~crappy.blocks.vision.CameraSource`, its ``config`` and
  ``allow_downstream_config`` arguments must therefore both be enabled. It is
  currently not possible to provide spot coordinates directly, so successful
  interactive configuration is mandatory.

  The configuration window owns the
  :class:`~crappy.tool.camera_config.config_tools.SpotsDetector` used for the
  initial detection. Once configuration is complete, this Block creates a
  :class:`~crappy.tool.image_processing.VideoExtensoTool`, which starts one
  independent
  :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker` Process
  per spot. Up to four spots can be tracked. With a single spot, only its
  position is meaningful and both strain values remain zero.

  For each processed image, the Block sends the image timestamp and metadata,
  the spot-center coordinates, and the vertical and horizontal strains through
  its regular output :class:`~crappy.links.Link` objects. The current spot
  boxes are additionally published under the reserved ``'overlay'`` label.
  They can be drawn by an :class:`~crappy.blocks.vision.ImageDisplayer` that
  receives both this regular Link and the images from the same source.

  This Block is similar to :class:`~crappy.blocks.DICVE`, which tracks textured
  patches instead of spots. :class:`~crappy.blocks.GPUVE` also performs
  video-extensometry, using GPU-accelerated image correlation.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               labels: str | Sequence[str] | None = None,
               raise_on_lost_spot: bool = True,
               white_spots: bool = False,
               update_thresh: bool = False,
               num_spots: int | None = None,
               safe_mode: bool = False,
               border: int = 5,
               min_area: int = 150,
               blur: int | None = 5,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 100) -> None:
    """Sets the spot-detection, tracking, and Block options.

    Args:
      labels: The five labels carrying the processing results. If omitted, they
        default to ``'t(s)'``, ``'meta'``, ``'Coord(px)'``, ``'Eyy(%)'``, and
        ``'Exx(%)'``. These respectively contain the image timestamp, its full
        metadata dictionary, a list of ``(y, x)`` spot-center coordinates, and
        the vertical and horizontal strains in percent. Custom labels must all
        be supplied at once in an iterable containing exactly five strings.
        The reserved ``'overlay'`` label is appended automatically and carries
        the current spot boxes.
      raise_on_lost_spot: If :obj:`True`, raises an exception when a spot is
        lost, which stops the test. If :obj:`False`, tracking stops but the test
        continues; the last valid result is sent once more with an empty
        overlay before this Block becomes idle.
      white_spots: If :obj:`True`, detects light spots over a dark background.
        If :obj:`False`, detects dark spots over a light background.
      update_thresh: If :obj:`True`, recalculates the gray-level detection
        threshold for every image. Keeping a fixed threshold generally yields
        less noisy measurements, while an adaptive threshold can make spots
        less likely to be lost under changing illumination. Adaptive
        thresholding may produce inconsistent results when spots are lost.
      num_spots: The number of spots to detect, from one to four. If provided,
        configuration attempts to find exactly that number and fails if too
        few spots are detected. If :obj:`None`, up to four spots are detected,
        potentially fewer.
      safe_mode: If :obj:`True`, considers overlapping spots lost immediately.
        Otherwise, first tries to eliminate the overlap by shrinking their
        detection windows. Enable this option when inconsistent measurements
        would have critical consequences.
      border: Number of pixels added on each side of the last known bounding
        box when searching for a spot in the next frame. It should exceed the
        expected spot displacement in pixels per frame. Excessively large
        values can allow noise or neighboring spots to hinder detection.
      min_area: Minimum area, in pixels, for an object to be considered a spot
        during initial detection. It should be adapted to the image resolution
        and the physical spot size.
      blur: Size of the median-blur kernel applied before spot detection. It
        must be a positive odd integer, or :obj:`None` to disable blurring. A
        slight blur can smooth image noise and improve detection at the cost of
        additional processing time.
      display_freq: If :obj:`True`, periodically displays the image-processing
        frequency.
      debug: If :obj:`True`, displays all log messages, including
        :obj:`~logging.DEBUG` messages. If :obj:`False`, only displays messages
        at :obj:`~logging.INFO` level or higher. If :obj:`None`, disables
        logging for this Block.
      freq: Target looping frequency for this Block. If :obj:`None`, loops as
        fast as possible. This is an upper bound for checking and processing
        newly received images, not the Camera acquisition frequency.
    """

    super().__init__(img_shape=None,
                     img_dtype=None,
                     display_freq=display_freq,
                     debug=debug,
                     freq=freq)

    # Forcing the labels into a list
    if labels is None:
      _labels: list[str] = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    elif isinstance(labels, str):
      _labels: list[str] = [labels]
    else:
      _labels: list[str] = list(labels)

    # Making sure a consistent number of labels was given
    if len(_labels) != 5:
      raise ValueError("The number of user-provided labels should be 5 !\n"
                       "Make sure that the time label was given")

    # Adding the reserved overlay label
    _labels.append('overlay')
    self.labels = _labels

    # Checking the validity of the provided arguments
    if not isinstance(raise_on_lost_spot, bool):
      raise TypeError("raise_on_lost_spot must be a boolean")
    if not isinstance(white_spots, bool):
      raise TypeError("white_spots must be a boolean")
    if (num_spots is not None and
        (not isinstance(num_spots, int) or not 0 < num_spots < 5)):
      raise ValueError("When provided, num_spots must be an integer between "
                       "1 and 4")
    if not isinstance(min_area, int) or min_area < 0:
      raise ValueError("min_area must be a positive integer")
    if (blur is not None and
        (not isinstance(blur, int) or blur < 1 or not blur % 2)):
      raise ValueError("When provided, blur must be a positive odd integer")
    if not isinstance(update_thresh, bool):
      raise TypeError("update_thresh must be a boolean")
    if not isinstance(safe_mode, bool):
      raise TypeError("safe_mode must be a boolean")
    if not isinstance(border, int) or border < 0:
      raise ValueError("border must be a positive integer")

    self._raise_on_lost_spot: bool = raise_on_lost_spot

    # Options forwarded to the configuration and processing layers
    self._white_spots: bool = white_spots
    self._num_spots: int | None = num_spots
    self._min_area: int = min_area
    self._blur: int | None = blur
    self._update_thresh: bool = update_thresh
    self._safe_mode: bool = safe_mode
    self._border: int = border

    # Attributes to set later on
    self._ve: VideoExtensoTool | None = None
    self._spots: SpotsBoxes | None = None
    self._thresh: int | None = None
    self._lost_spots: bool = False

    self._last_data: tuple[list[tuple[float | int, ...]],
                           float | int, float | int] | None = None

  def prepare(self) -> None:
    """Receives the source configuration and starts the spot trackers.

    This method checks that the Block has exactly one input ImageLink, no input
    regular Link, and no output ImageLink. It then waits for the upstream
    configuration result, initializes the VideoExtenso tool with the selected
    spots and gray-level threshold, retrieves the shared image buffers, and
    starts one Tracker Process for each selected spot.

    Raises:
      IOError: If the Block's Link topology is unsupported.
      RuntimeError: If no usable configuration was received or required
        startup objects are unavailable.
      NotImplementedError: If more than one source returns configuration data.
    """

    # Ensuring Link consistency
    if not self.img_inputs:
      raise IOError("This VisionBlock is useless without an input ImageLink")
    if len(self.img_inputs) > 1:
      raise IOError("This VisionBlock accepts exactly one input ImageLink")
    if self.img_outputs:
      raise IOError("This VisionBlock does not support output ImageLink")
    if self.inputs:
      raise IOError("This Block does not accept input Links")

    # Receive the configuration information from upstream Blocks
    configs = self.recv_configs()
    valid = [config for config in configs.values() if config is not None]
    if not len(valid):
      raise RuntimeError("No configuration information received from upstream "
                         "Blocks, cannot proceed")
    elif len(valid) > 1:
      raise NotImplementedError("Ambiguous situation with at least two "
                                "configurations received from upstream "
                                "Blocks, don't know how to handle")
    for source, config in configs.items():
      if config is not None:
        try:
          self._spots, self._thresh = config
        except (ValueError, TypeError):
          self.log(logging.ERROR, f"Got invalid configuration data from "
                                  f"Block {source}")
          raise

    # Catch uninitialized attributes early
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

    # Mandatory otherwise the Block won't run
    super().prepare()

    if self._ve is None:
      raise RuntimeError("The VideoExtensoTool isn't initialized")
    self.log(logging.INFO, "Starting the VideoExtenso spot tracker processes")
    self._ve.start_tracking()

  def loop(self) -> None:
    """Processes the newest image and sends strain data and spot overlays.

    If no new image is available, the method returns without processing. For a
    new image, the spot Trackers update their bounding boxes and the
    VideoExtenso tool calculates strain from the distance between the extreme
    spot centers relative to their configured initial distance.

    When spot loss is tolerated, the final valid measurements are sent once
    more with an empty overlay iterable, then the Block remains idle. This
    preserves consistent Link labels while clearing the control display.
    """

    # If the spots were lost but raise_on_lost_spot is False, do nothing
    if self._lost_spots:
      self.log(logging.DEBUG, "Spots were lost, remaining idle")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return

    # Nothing to do if no new image was received
    if not (upd_links := self.receive_imgs()):
      self.log(logging.DEBUG, "No new image received during this loop")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Get the ImageLink name
    upd_link, = upd_links

    # Handles to the received data
    metadata = self.last_received[upd_link].metadata
    if metadata is None:
      raise RuntimeError("At that point, the image metadata should not be "
                         "empty")
    img = self.last_received[upd_link].img

    # Processing the received frame
    try:
      self.log(logging.DEBUG, "Processing the received image")

      if self._ve is None:
        raise RuntimeError("The VideoExtensoTool isn't initialized")

      # Sending the results to the downstream Blocks, including the overlay
      if (data := self._ve.get_data(img)) is not None:
        self.send([metadata['t(s)'], metadata, *data, self._ve.spots])

        # Save the last data points for the case when spots are lost
        self._last_data = data

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
        # Send a message with the last data points and no overlay for stopping
        # the display of the overlay
        if self._last_data is not None:
          self.send([metadata['t(s)'], metadata, *self._last_data, list()])

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def finish(self) -> None:
    """Stops the spot trackers and releases inherited image resources."""

    try:
      if self._ve is not None:
        self.log(logging.INFO, "Stopping the spot trackers before returning")
        self._ve.stop_tracking()
    finally:
      super().finish()

  def request_config(self, source: str) -> ConfigRequest:
    """Builds the VideoExtenso configuration request for an image source.

    Args:
      source: Name of the upstream image source that should run the
        configuration window.

    Returns:
      A request for :class:`~crappy.tool.camera_config.VideoExtensoConfig`
      containing the spot-detection options set on this Block.
    """

    return ConfigRequest(requester=self.name,
                         args=tuple(),
                         kwargs={'white_spots': self._white_spots,
                                 'num_spots': self._num_spots,
                                 'min_area': self._min_area,
                                 'blur': self._blur,
                                 'update_thresh': self._update_thresh,
                                 'safe_mode': self._safe_mode,
                                 'border': self._border},
                         configurator=VideoExtensoConfig,
                         img_source=source)
