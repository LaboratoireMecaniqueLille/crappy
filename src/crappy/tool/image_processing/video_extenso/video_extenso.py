# coding: utf-8

from multiprocessing import (Pipe, current_process, connection, Event,
                             Queue as MPQueue)
from multiprocessing import synchronize
from threading import Thread, Event as ThreadEvent
from multiprocessing.queues import Queue
import numpy as np
from itertools import combinations
from time import monotonic
import logging
import logging.handlers
from select import select
from platform import system
from pickle import dumps

from ...camera_config import SpotsBoxes, Box
from .tracker import Tracker
from ...._failure import FailureManager
from ...._exceptions import (ProcessExitError, ShutdownTimeoutError,
                             StopRequested)


class VideoExtensoTool:
  """This class is the core of the :class:`~crappy.blocks.VideoExtenso` Block.

  It performs spot tracking on up to `4` spots on the images acquired by the
  :class:`~crappy.camera.meta_camera.camera.Camera`, and computes the strain
  values at each new image. For each spot, the tracking is performed by an
  independent
  :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker` Process.

  This tool is created by
  :class:`~crappy.blocks.camera_processes.VideoExtensoProcess` and owns the
  creation, communication, and shutdown of those Tracker processes; the public
  :class:`~crappy.blocks.VideoExtenso` Block does not manage them directly.

  It is possible to track only one spot, in which case only the position of its
  center is returned and the strain values are left to `0`.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10 renamed from *Video_extenso* to *VideoExtenso*
  .. versionchanged:: 2.0.0 renamed from *VideoExtenso* to *VideoExtensoTool*
  """

  def __init__(self,
               spots: SpotsBoxes,
               thresh: int,
               log_level: int | None,
               log_queue: Queue,
               white_spots: bool,
               update_thresh: bool,
               safe_mode: bool,
               border: int,
               blur: int | None,
               failure_manager: FailureManager,
               stop_event: synchronize.Event,
               raise_on_lost_spot: bool = False) -> None:
    """Sets the arguments and the other instance attributes.

    Args:
      spots: An instance of the
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` tool
        containing the coordinates of the spots to track.

        .. versionadded:: 2.0.0
      thresh: The grey level value of the threshold to use for discriminating
        spots from the background, as an :obj:`int`. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      white_spots: If :obj:`True`, detects white objects over a black
        background, else black objects over a white background. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      update_thresh: If :obj:`True`, the gray level threshold for detecting the
        spots is re-calculated at each new image. Otherwise, the first
        calculated threshold is kept for the entire test. The spots are less
        likely to be lost with adaptive threshold, but the measurement will be
        more noisy. Adaptive threshold may also yield inconsistent results when
        spots are lost. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      safe_mode: If :obj:`True`, the class returns a spot-loss reason string as
        soon as overlapping spots are detected. Otherwise, it will first try to
        reduce the detection window to get rid of overlapping. This argument
        should be used when inconsistency in the results may have critical
        consequences.
      border: When searching for the new position of a spot, the class will
        search in the last known bounding box of this spot plus a few
        additional pixels in each direction. This argument sets the number of
        additional pixels to use. It should be greater than the expected
        "speed" of the spots, in pixels / frame. But if it's set too high,
        noise or other spots might hinder the detection. Passed to the
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      blur: The size in pixels (as an odd :obj:`int` greater than `1`) of the
        kernel to use when applying a median blur filter to the image before
        the spot detection. If not given, no blurring is performed. A slight 
        blur improves the spot detection by smoothening the noise, but also 
        takes a bit more time compared to no blurring. Passed to the 
        :class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker`
        and not used in this class.
      failure_manager: The :class:`~crappy._failure.FailureManager` to use for
        reporting caught exceptions.

        .. versionadded:: 2.2.0
      stop_event: :obj:`~multiprocessing.Event` to set when a failure requires
        the owning Block to stop.

        .. versionadded:: 2.2.0
      raise_on_lost_spot: If :obj:`True`, report a lost spot as a failure.
        If :obj:`False`, return a reason string from :meth:`get_data` so the
        caller can decide how to respond.

    .. versionremoved:: 2.0.0 *num_spots* and *min_area* arguments
    """

    # Set up the FailureManager and logger
    name = f'{current_process().name}.{type(self).__name__}'
    self._logger = logging.getLogger(name)
    self._logger.disabled = log_level is None
    if log_level is not None:
      self._logger.setLevel(log_level)
    self._failure_manager = failure_manager

    # Attributes for handling and reporting failures
    self._owner_stop: synchronize.Event = stop_event
    self._raise_on_lost_spot: bool = raise_on_lost_spot
    self._failure_queues: list[Queue] = list()
    self._failure_thread: Thread | None = None
    self._failure_thread_stop: ThreadEvent | None = None
    self._tracking_stop: synchronize.Event | None = None
    self._unexpected_exits: set[Tracker] = set()

    # These attributes will be used later
    self._consecutive_overlaps: int = 0
    self._trackers: list[Tracker] = list()
    self._started_trackers: list[Tracker] = list()
    self._pipes: list[connection.Connection] = list()
    self._tracker_pipes: list[connection.Connection] = list()

    # Setting the args
    self._white_spots: bool = white_spots
    self._update_thresh: bool = update_thresh
    self._safe_mode: bool = safe_mode
    self._border: int = border
    self._blur: int | None = blur
    self.spots: SpotsBoxes = spots
    self._thresh: int = thresh

    self._log_level: int | None = log_level
    self._log_queue: Queue = log_queue

    self._last_warn: float = monotonic()
    self._system: str = system()

  def __del__(self) -> None:
    """Tries to stop tracking when the tool is discarded."""

    # Case of partly initialized object
    if not hasattr(self, '_trackers') or not hasattr(self, '_pipes'):
      return

    # Make sure that the tracking is stopped and resources are released
    with self._failure_manager.collect('Tracker destructor',
                                       raise_on_exit=False):
      with self._failure_manager.step('stop tracking'):
        self.stop_tracking()

  def start_tracking(self) -> None:
    """Starts tracking the selected spots.

    Raises:
      AttributeError: No spots were selected.
      RuntimeError: Tracking has already started.
    """

    if self.spots.empty():
      raise AttributeError("No spots selected, aborting !")
    if self._trackers:
      raise RuntimeError("Tracking has already started")

    # Initialize synchronization objects
    self._tracking_stop = Event()
    self._unexpected_exits.clear()

    for spot in self.spots:
      if spot is None:
        continue

      inlet, outlet = Pipe()
      failure_queue = MPQueue()
      # Retain all endpoints for cleanup later
      self._pipes.append(inlet)
      self._tracker_pipes.append(outlet)
      self._failure_queues.append(failure_queue)
      tracker = Tracker(pipe=outlet,
                        logger_name=f"{current_process().name}."
                                    f"{type(self).__name__}",
                        log_level=self._log_level,
                        log_queue=self._log_queue,
                        white_spots=self._white_spots,
                        thresh=None if self._update_thresh else self._thresh,
                        blur=self._blur,
                        failure_queue=failure_queue,
                        stop_event=self._tracking_stop,
                        block_name=self._failure_manager.block_name,
                        raise_on_lost_spot=self._raise_on_lost_spot)
      self._trackers.append(tracker)
      tracker.start()
      self._started_trackers.append(tracker)
      # The child owns its copy of this endpoint, it's no longer needed here
      outlet.close()
      self._tracker_pipes.remove(outlet)

    # Start a Thread for handling al the incoming Exceptions
    if self._failure_queues:
      self._failure_thread_stop = ThreadEvent()
      self._failure_thread = Thread(target=self._receive_failures,
                                    name='trackers.failure-reader',
                                    daemon=True)
      self._failure_thread.start()

  def stop_tracking(self) -> None:
    """Stops tracking and releases its resources.

    Processes that do not stop in time are forcibly stopped. Cleanup failures
    are reported together after all cleanup operations have been attempted.

    Raises:
      BaseExceptionGroup: Failures encountered while stopping.
    """

    manager = self._failure_manager
    with manager.collect('VideoExtensoTool cleanup'):
      with manager.step('start report receiver'):
        if (self._failure_queues and
            (self._failure_thread is None or
             self._failure_thread.ident is None)):
          # Make sure there is an Exception Thread running for cleanup
          self._failure_thread_stop = ThreadEvent()
          self._failure_thread = Thread(target=self._receive_failures,
                                        name='trackers.failure-reader',
                                        daemon=True)
          self._failure_thread.start()

      # Make sure that the Trackers have received the stop signal
      with manager.step('request tracker shutdown'):
        if self._tracking_stop is not None:
          self._tracking_stop.set()

      trackers = tuple(self._trackers)
      started = [tracker for tracker in trackers
                 if tracker in self._started_trackers]
      forced = set()
      exits = list()

      # The deadline covers termination of all the Workers together
      deadline = monotonic() + 3.0
      for tracker in started:
        with manager.step(f'join {tracker.name}'):
          tracker.join(max(0.0, deadline - monotonic()))

      # Check which Workers are still alive after 3 seconds
      survivors = list()
      for tracker in started:
        name = tracker.name
        alive = True
        with manager.step(f'check {name}'):
          alive = tracker.is_alive()
        if alive:
          with manager.step(f'shutdown deadline for {name}'):
            raise ShutdownTimeoutError(f"Tracker {name} did not stop within "
                                       f"3 seconds",
                                       process_name=name,
                                       pid=tracker.pid,
                                       phase='tracker cleanup',
                                       timeout=3.0)
          with manager.step(f'terminate {name}'):
            tracker.terminate()
            forced.add(tracker)
          survivors.append(tracker)

      # Give all surviving Workers 0.5 seconds to terminate, and kill survivors
      deadline = monotonic() + 0.5
      killed = list()
      for tracker in survivors:
        name = tracker.name
        with manager.step(f'join terminated {name}'):
          tracker.join(max(0.0, deadline - monotonic()))
        alive = True
        with manager.step(f'check terminated {name}'):
          alive = tracker.is_alive()
        if alive:
          with manager.step(f'kill {name}'):
            tracker.kill()
            forced.add(tracker)
          killed.append(tracker)

      # Give all surviving Workers 0.5 seconds to get killed
      deadline = monotonic() + 0.5
      for tracker in killed:
        with manager.step(f'join killed {tracker.name}'):
          tracker.join(max(0.0, deadline - monotonic()))

      for tracker in trackers:
        name = tracker.name
        if tracker in self._started_trackers:
          with manager.step(f'observe exit of {name}'):
            code = tracker.exitcode
            if isinstance(code, int):
              exits.append((name, tracker.pid, code,
                            tracker in self._unexpected_exits,
                            tracker in forced))
        # Properly close all the monitored Trackers
        with manager.step(f'close process {name}'):
          tracker.close()
          self._trackers.remove(tracker)
          if tracker in self._started_trackers:
            self._started_trackers.remove(tracker)

      # Close all the instantiated Pipes
      for role, pipes in (('parent', self._pipes),
                          ('child', self._tracker_pipes)):
        for index, pipe in enumerate(tuple(pipes)):
          with manager.step(f'close {role} Tracker Pipe {index + 1}'):
            pipe.close()
            pipes.remove(pipe)

      # Close the Exception Thread and report it if it didn't stop properly
      with manager.step('stop tracker receiver'):
        if self._failure_thread_stop is not None:
          self._failure_thread_stop.set()
      with manager.step('join tracker receiver'):
        if (self._failure_thread is not None and
            self._failure_thread.ident is not None):
          self._failure_thread.join(0.5)
          if self._failure_thread.is_alive():
            raise ShutdownTimeoutError("Tracker failure receiver did not stop",
                                       phase='monitoring', timeout=0.5)
          self._failure_thread = None

      # Close all the opened Queues
      for queue in tuple(self._failure_queues):
        with manager.step('close tracker failure Queue'):
          queue.close()
          self._failure_queues.remove(queue)

      # Report Trackers whose exit code is unexpected
      reported = {(report.process_name, report.pid)
                  for report, _ in manager.snapshot()}
      reported.update((error.process_name, error.pid)
                      for _, error in manager.snapshot()
                      if isinstance(error, ProcessExitError))
      for name, pid, code, early, terminated in exits:
        with manager.step(f'verify exit of {name}'):
          if ((name, pid) not in reported and not terminated and
              (code != 0 or early)):
            raise ProcessExitError(f"Tracker {name} exited unexpectedly "
                                   f"({code})",
                                   process_name=name,
                                   pid=pid,
                                   phase='tracking',
                                   exitcode=code)

  def get_data(self,
               img: np.ndarray
               ) -> tuple[list[tuple[float, ...]], float, float] | str | None:
    """Takes an image as an input, performs spot detection on it, computes the
    strain from the newly detected spots, and returns the spot positions and
    strain values.

    Args:
      img: The image on which the spots should be detected.

    Returns:
      A reason string when tracking can no longer continue normally. Otherwise,
      a :obj:`list` containing :obj:`tuple` with the coordinates of the centers
      of the detected spots, and the calculated x and y strain values.

    Raises:
      BaseExceptionGroup: Unexpected tracker or communication failures.
      RuntimeError: If the Tracker processes were not started correctly, or
        if one of them exited without reporting a lost spot.
    
    .. versionchanged:: 1.5.10 renamed from *get_def* to *get_data*
    """

    manager = self._failure_manager
    manager.raise_if_failed('Tracking failures')
    if self._owner_stop.is_set():
      raise StopRequested

    spots = [(i, spot) for i, spot in enumerate(self.spots)
             if spot is not None]

    # Make sure one Tracker and Pipe were started for each configured spot
    if (not self._trackers or
        len(self._trackers) != len(spots) or
        len(self._pipes) != len(spots)):
      raise RuntimeError("The spot Trackers were not started correctly, "
                         "cannot process image")

    # Sending the latest sub-image containing the spot to track
    # Also sending the coordinates of the top left pixel
    communication_errors = list()
    for pipe, (_, spot) in zip(self._pipes, spots):
      x_top, x_bottom, y_left, y_right = spot.sorted()
      slice_y = slice(max(0, y_left - self._border),
                      min(img.shape[0], y_right + self._border))
      slice_x = slice(max(0, x_top - self._border),
                      min(img.shape[1], x_bottom + self._border))
      try:
        self._send(pipe, (slice_y.start, slice_x.start,
                          img[slice_y, slice_x]))
      except (EOFError, OSError) as error:
        communication_errors.append(error)

    # Receiving the latest available data from each Tracker
    for pipe, (i, _) in zip(self._pipes, spots):
      box = None
      lost_spot = None
      try:
        if not pipe.poll(timeout=0.1):
          continue
        box = pipe.recv()
        # Means there was a tracking or communication error
        if isinstance(box, str):
          lost_spot = box
        while pipe.poll():
          box = pipe.recv()
          # Means there was a tracking or communication error
          if isinstance(box, str):
            lost_spot = box
      except (EOFError, OSError) as error:
        # Keep a response received just before the peer closed its connection.
        communication_errors.append(error)

      if lost_spot is not None:
        return lost_spot

      if box is not None:
        self.spots[i] = box

    # If there's any dead Tracker or failure, stop tracking and report
    dead = [tracker for tracker in self._trackers if not tracker.is_alive()]
    if communication_errors or dead:
      self._unexpected_exits.update(dead)
      with manager.collect('Tracking communication', raise_on_exit=False):
        with manager.step('stop tracking'):
          self.stop_tracking()
        if not manager.snapshot():
          for error in communication_errors:
            manager.record_context_exception(error, 'image Pipe')
      manager.raise_if_failed('Tracking failures')

    overlap = False

    # Checking if the newly received boxes overlaps with each other
    for box_1, box_2 in combinations(self.spots, 2):

      if box_1 is None or box_2 is None:
        continue

      if self._overlap_box(box_1, box_2):

        # If there's overlapping in safe mode, reporting spot loss directly
        if self._safe_mode:
          return "Overlapping spots detected in safe mode"

        self._log(logging.WARNING, "Overlapping detected ! Reducing spot "
                                   "window")

        # If we're not in safe mode, simply reduce the boxes by 1 pixel
        # Also, make sure the box is not being reduced too much
        overlap = True
        x_top_1, x_bottom_1, y_left_1, y_right_1 = box_1.sorted()
        x_top_2, x_bottom_2, y_left_2, y_right_2 = box_2.sorted()

        box_1.x_start = min(x_top_1 + 1, int(box_1.x_centroid - 2))
        box_1.y_start = min(y_left_1 + 1, int(box_1.y_centroid - 2))
        box_1.x_end = max(x_bottom_1 - 1, int(box_1.x_centroid + 2))
        box_1.y_end = max(y_right_1 - 1, int(box_1.y_centroid + 2))

        box_2.x_start = min(x_top_2 + 1, int(box_2.x_centroid - 2))
        box_2.y_start = min(y_left_2 + 1, int(box_2.y_centroid - 2))
        box_2.x_end = max(x_bottom_2 - 1, int(box_2.x_centroid + 2))
        box_2.y_end = max(y_right_2 - 1, int(box_2.y_centroid + 2))

    if overlap:
      self._consecutive_overlaps += 1
      if self._consecutive_overlaps > 10:
        return "Too many consecutive overlaps"
    else:
      self._consecutive_overlaps = 0

    # If there are multiple spots, the x and y strains can be computed
    if len(self.spots) > 1:
      x = [spot.x_centroid for spot in self.spots if spot is not None]
      y = [spot.y_centroid for spot in self.spots if spot is not None]
      # Double check that there's no None value
      x_len, y_len = len(x), len(y)
      x = [el for el in x if el is not None]
      y = [el for el in y if el is not None]
      if x_len != len(x):
        raise RuntimeError("One of the spot's x centroid wasn't computed as "
                           "expected")
      if y_len != len(y):
        raise RuntimeError("One of the spot's y centroid wasn't computed as "
                           "expected")
      # The strain is calculated based on the positions of the extreme
      # spots in each direction
      try:
        exx = ((max(x) - min(x)) / self.spots.x_l0 - 1) * 100
      except ZeroDivisionError:
        exx = 0
      try:
        eyy = ((max(y) - min(y)) / self.spots.y_l0 - 1) * 100
      except ZeroDivisionError:
        eyy = 0
      centers = list(zip(y, x))
      return centers, eyy, exx

    # If only one spot was detected, the strain isn't computed
    else:
      if (spot := self.spots[0]) is None:
        raise RuntimeError("All the spots are None, this should not be")
      x = spot.x_centroid
      y = spot.y_centroid
      if x is None or y is None:
        raise RuntimeError("Cannot return the spot centroid, it was never "
                           "computed!")
      return [(y, x)], 0.0, 0.0

  def _log(self, level: int, msg: str) -> None:
    """Wrapper for recording log messages.

    Args:
      level: The logging level of the message, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    self._logger.log(level, msg)

  def _send(self,
            conn: connection.Connection,
            val: tuple[int, int, np.ndarray]) -> None:
    """Sends an image and its coordinates to a Tracker process.

    In Linux, checks that the Pipe is not full before sending the message.

    Args:
      conn: The Connection to use for sending the message.
      val: The message to send to the Tracker process.
    """

    if self._system == 'Linux':
      if select(list(), [conn], list(), 0)[1]:
        conn.send_bytes(dumps(val))
      elif monotonic() - self._last_warn > 1:
          self._last_warn = monotonic()
          self._log(logging.WARNING, f"Cannot send the image to process to the"
                                     f" Tracker process, the Pipe is full !")
    else:
      conn.send_bytes(dumps(val))

  def _receive_failures(self) -> None:
    """Receives tracker failures and requests the owning Block to stop.

    .. versionadded:: 2.2.0
    """

    # Keep references in case the class is reset
    manager = self._failure_manager
    incoming = tuple(self._failure_queues)
    stop = self._failure_thread_stop
    tracking_stop = self._tracking_stop
    owner_stop = self._owner_stop
    if stop is None:
      return

    with manager.collect('Tracker reception', raise_on_exit=False):
      with manager.step('receive reports'):
        notified = False
        while True:
          received = manager.receive(incoming)
          # If a failure report was received, means we have to stop
          if manager.snapshot() and not notified:
            with manager.step('request tracker stop'):
              if tracking_stop is not None:
                tracking_stop.set()
            with manager.step('request owner stop'):
              owner_stop.set()
            notified = True
          # Only stop once there are no more reports to receive
          if stop.is_set() and not received:
            break
          if not received:
            stop.wait(0.05)

  @staticmethod
  def _overlap_box(box_1: Box, box_2: Box) -> bool:
    """Determines whether two boxes are overlapping or not."""

    x_min_1, x_max_1, y_min_1, y_max_1 = box_1.sorted()
    x_min_2, x_max_2, y_min_2, y_max_2 = box_2.sorted()

    return max((min(x_max_1, x_max_2) - max(x_min_1, x_min_2)), 0) * max(
      (min(y_max_1, y_max_2) - max(y_min_1, y_min_2)), 0) > 0

  @staticmethod
  def _overlap_bbox(prop_1, prop_2) -> bool:
    """Determines whether two bboxes are overlapping or not."""

    y_min_1, x_min_1, y_max_1, x_max_1 = prop_1.bbox
    y_min_2, x_min_2, y_max_2, x_max_2 = prop_2.bbox

    return (max((int(min(x_max_1, x_max_2)) - int(max(x_min_1, x_min_2))), 0) *
            max((int(min(y_max_1, y_max_2)) - int(max(y_min_1, y_min_2))), 0)
            > 0)
