# coding: utf-8

from multiprocessing import Process, managers, get_start_method, \
  current_process
from multiprocessing.synchronize import Event, RLock, Barrier
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import BrokenBarrierError
import numpy as np
from typing import Optional, Union, Any
from collections.abc import Iterable
import logging
import logging.handlers
from select import select
from time import time
from platform import system

from ...links import Link
from ..._global import LinkDataError
from ...tool.camera_config import Overlay


class CameraProcess(Process):
  """This :obj:`~multiprocessing.Process` is the base class for all the helper
  Processes of the :class:`~crappy.blocks.Camera` Block.
  
  It defines a base architecture that all the children classes use, and on top
  of which they add the specific actions they perform. This class is not meant 
  to be instantiated as is, it should always be subclassed.
  
  This class and its children can be seen as a tool for the 
  :class:`~crappy.blocks.Camera` Block and its children. Several instances can
  be started by the Camera Block, each instance performing a different action
  with the acquired images. This allows to parallelize image acquisition, 
  display, processing and recording on multiple CPU cores, to increase the 
  achieved FPS.
  
  The Camera Block performs the acquisition, and makes the latest captured
  image available to all the CameraProcess :obj:`~multiprocessing.Process` 
  through an :obj:`~multiprocessing.Array`. They are then free to run at their
  own rhythm, but are sure to always grab the latest available frame.
  
  The instantiation, startup and termination of the CameraProcesses is all
  managed by the parent :class:`~crappy.blocks.Camera` Block, depending on the
  provided arguments. Users should normally not need to call this class
  themselves.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self) -> None:
    """Initializes the parent class and all the instance attributes."""

    super().__init__()
    self.name = f"{current_process().name}.{type(self).__name__}"
    self._system = system()

    # Logging-related objects
    self._log_queue: Optional[Queue] = None
    self._logger: Optional[logging.Logger] = None
    self._log_level: Optional[int] = None

    # These objects will be shared later by the Camera Block
    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._cam_barrier: Optional[Barrier] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[tuple[int, int]] = None
    self._to_draw_conn: Optional[Connection] = None
    self._outputs: list[Link] = list()
    self._labels: list[str] = list()
    self.img: Optional[np.ndarray] = None
    self._dtype = None
    self.metadata = {'ImageUniqueID': None}
    self._img0_set = False

    # Other attribute for internal use
    self._last_warn = time()
    self.fps_count = 0
    self._display_freq: Optional[bool] = None
    self._last_fps = time()

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
                 barrier: Barrier,
                 event: Event,
                 shape: Union[tuple[int, int], tuple[int, int, int]],
                 dtype,
                 to_draw_conn: Optional[Connection],
                 outputs: list[Link],
                 labels: Optional[list[str]],
                 log_queue: Queue,
                 log_level: Optional[int] = 20,
                 display_freq: bool = False) -> None:
    """Method allowing the :class:`~crappy.blocks.Camera` Block to share
    :mod:`multiprocessing` synchronization objects with this class.
    
    Args:
      array: The :obj:`~multiprocessing.Array` containing the last frame
        acquired by the Camera Block.
      data_dict: A :obj:`dict` managed by a :obj:`~multiprocessing.Manager` and
        containing the metadata of the last acquired frame.
      lock: A :obj:`~multiprocessing.RLock` ensuring that the CameraProcess and
        the Camera Block do not try to access the shared array at the same
        time.
      barrier: A :obj:`~multiprocessing.Barrier` ensuring that all the
        CameraProcesses wait for a start signal from the Camera Block before
        starting to run.
      event: A :obj:`~multiprocessing.Event` indicating to the CameraProcess
        when to stop running. It is either set by the Camera Block, or by a
        CameraProcess.
      shape: The expected shape of the image, as a :obj:`tuple`. It is 
        necessary as the frames are shared as a one-dimensional array.
      dtype: The expected dtype of the image. It is necessary for 
        reconstructing the image from the one-dimensional shared array.
      to_draw_conn: A :obj:`~multiprocessing.Connection` for sending or
        receiving :class:`~crappy.tool.camera_config.config_tools.Overlay`
        objects to draw on top of the displayed image.
      outputs: The :class:`~crappy.links.Link` objects for sending data to
        downstream Blocks. They are the same as those owned by the Camera 
        Block.
      labels: The labels to use when sending data to downstream Blocks.
      log_queue: A :obj:`~multiprocessing.Queue` for sending the log messages
        to the main :obj:`~logging.Logger`, only used in Windows.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
      display_freq: If :obj:`True`, the looping frequency of this class will be
        displayed while running.
    """

    self._img_array = array
    self._data_dict = data_dict
    self._lock = lock
    self._cam_barrier = barrier
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype
    self._to_draw_conn = to_draw_conn
    self._outputs = outputs
    self._labels = labels

    # Logging related attributes
    self._log_queue = log_queue
    self._log_level = log_level
    self._display_freq = display_freq

    self.img = np.empty(shape=shape, dtype=dtype)

  def run(self) -> None:
    """This method is the core of the :obj:`~multiprocessing.Process`.
    
    It starts by initializing the :obj:`~logging.Logger`, and then performs any
    additional action required before processing images. Once all the
    CameraProcesses are ready, it loops forever and processes images until told
    to stop or an exception is raised. And finally, it performs any action
    required for properly exiting.

    This method is quite similar to the :meth:`~crappy.blocks.Block.run` method
    of the :class:`~crappy.blocks.Block`, although it is much simpler.
    """

    try:
      # First thing, setting the Logger
      self._set_logger()
      self.log(logging.INFO, "Logger configured")

      # Initializing the CameraProcess, and breaking the Barrier to warn the
      # other CameraProcesses in case something goes wrong
      try:
        self.init()
      except (Exception,):
        self._cam_barrier.abort()
        self.log(logging.ERROR, "Breaking the barrier due to caught exception"
                                " while preparing")
        raise

      # Waiting for all other CameraProcess to be ready
      self.log(logging.INFO, "Waiting for the other Camera processes to be "
                             "ready")
      self._cam_barrier.wait()
      self.log(logging.INFO, "All Camera processes ready now")

      self._last_fps = time()

      # Looping forever until told to stop or an exception is raised
      while not self._stop_event.is_set():
        # Only looping if a new image is available
        if self._get_data():
          self.log(logging.DEBUG, "Running the loop method")
          self.loop()
          self.fps_count += 1

        # Displaying the looping frequency is required
        if self._display_freq:
          t = time()
          if t - self._last_fps > 2:
            self.log(logging.INFO, f"Images processed /s: "
                                   f"{self.fps_count / (t - self._last_fps)}")
            self._last_fps = t
            self.fps_count = 0

      self.log(logging.INFO, "Stop event set, stopping the processing")

    # Case when CTRL+C was pressed
    except KeyboardInterrupt:
      self.log(logging.INFO, "KeyboardInterrupt caught, stopping the "
                             "processing")

    # Case when another CameraProcess raised an exception while initializing
    except BrokenBarrierError:
      self.log(logging.WARNING,
               "Exception raised in another Camera process while waiting "
               "for all Camera processes to be ready, stopping")

    # Handling any other unexpected exception
    except (Exception,) as exc:
      self._logger.exception("Exception caught wile running !", exc_info=exc)
      self.log(logging.ERROR, "Setting the stop event to stop the other "
                              "Camera processes")
      self._stop_event.set()
      raise

    # Always calling finish in the end
    finally:
      self.finish()

  def init(self) -> None:
    """This method should perform any action required for initializing the
    CameraProcess.

    It is meant to be overwritten by children classes, at is otherwise does not
    perform any action.
    
    It is called right after the :obj:`~multiprocessing.Process` starts, and
    when the images haven't started to be acquired yet.
    """

    ...

  def loop(self) -> None:
    """This method is the main loop of the CameraProcess.
    
    It is called repeatedly until the :obj:`~multiprocessing.Process` is told
    to stop. It should perform the desired action for handling the latest
    available frame, stored in the *self.img* attribute. The latest available
    metadata containing at least the timestamp and frame index of the latest
    image is stored in *self.metadata*.
    
    This method is meant to be overwritten by children classes, at is otherwise 
    does not perform any action.
    """

    ...

  def finish(self) -> None:
    """This method should perform any action required for properly exiting the
    CameraProcess.

    It is meant to be overwritten by children classes, at is otherwise does not
    perform any action.

    It is the last method called before the :obj:`~multiprocessing.Process`
    ends, and at that point no more images are being acquired.
    """

    ...

  def send(self, data: Optional[Union[dict[str, Any],
                                      Iterable[Any]]]) -> None:
    """This method allows sending data to downstream Blocks.

    It is similar to the :meth:`~crappy.blocks.Block.send` method of the
    :class:`~crappy.blocks.Block`. It accepts data either as a :obj:`dict`, or
    as an iterable of values.
    """

    # Just in case, not handling non-existing data
    if data is None:
      return

    # Case when the data to send is not given as a dict
    if not isinstance(data, dict):
      # First, checking that labels are provided
      if self._labels is None or not self._labels:
        self._logger.log(logging.ERROR, "Trying to send data as an iterable, "
                                        "but no labels are specified !")
        raise LinkDataError

      # Trying to convert iterable data to dict using the given labels
      try:
        self._logger.log(logging.DEBUG, f"Converting {data} to dict before "
                                        f"sending")
        data = dict(zip(self._labels, data))
      except TypeError:
        self._logger.log(logging.ERROR, f"Cannot convert data to send (of type"
                                        f" {type(data)}) to dict ! Please "
                                        f"ensure that the data is given as an "
                                        f"iterable, as well as the labels.")

    # Sending the data to the downstream Blocks
    for link in self._outputs:
      self._logger.log(logging.DEBUG, f"Sending {data} to Link {link.name}")
      link.send(data)

  def send_to_draw(self, to_draw: Iterable[Overlay]) -> None:
    """This method sends a collection of
    :class:`~crappy.tool.camera_config.config_tools.Overlay` objects to the
    :class:`~crappy.blocks.camera_processes.Displayer` CameraProcess.

    The overlays are sent by the CameraProcess performing the image processing,
    so that the area(s) of interest can be displayed simultaneously.
    """

    # Not sending if there's no Connection to send data through
    if self._to_draw_conn is None:
      return

    self.log(logging.DEBUG, "Sending the overlays to the displayer process")

    # Sending the overlay
    if self._system == 'Linux':
      if select([], [self._to_draw_conn], [], 0)[1]:
        # Can only check on Linux if a pipe is full
        self._to_draw_conn.send(to_draw)
      elif time() - self._last_warn > 1:
        # Warning in case the pipe is full
        self._last_warn = time()
        self.log(logging.WARNING, f"Cannot send the overlay to draw to the "
                                  f"Displayer process, the Pipe is full !")
    else:
      self._to_draw_conn.send(to_draw)

  def log(self, level: int, msg: str) -> None:
    """Sends a log message to the :obj:`~logging.Logger`.

    Args:
      level: The logging level, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(level, msg)

  def _get_data(self) -> bool:
    """This method allows to grab the latest available frame.

    It first acquired the :obj:`~multiprocessing.RLock` protecting the shared
    :obj:`~multiprocessing.Array` containing the image, then copies the image
    locally and releases the Lock. It also copies the metadata associated to
    the image.

    Returns:
      :obj:`True` in case a frame was acquired and needs to be handled, or
      :obj:`False` if no frame was grabbed and nothing should be done.
    """

    # Acquiring the Lock to avoid conflicts with other CameraProcesses
    with self._lock:

      # In case there's no frame grabbed yet
      if 'ImageUniqueID' not in self._data_dict:
        return False

      # In case the frame in buffer was already handled during a previous loop
      if self._data_dict['ImageUniqueID'] == self.metadata['ImageUniqueID']:
        return False

      # Copying the metadata
      self.metadata = self._data_dict.copy()

      self.log(logging.DEBUG, f"Got new image to process with id "
                              f"{self.metadata['ImageUniqueID']}")

      # Copying the frame
      np.copyto(self.img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def _set_logger(self) -> None:
    """Initializes the :obj:`~logging.Logger` for the CameraProcess.

    If :func:`multiprocessing.get_start_method` is `'spawn'` (mostly Windows
    and Mac for Python < 3.12), redirects the log messages to a
    :obj:`multiprocessing.Queue` sending them to the main Process.
    """

    logger = logging.getLogger(self.name)

    # Disabling logging if requested
    if self._log_level is not None:
      logger.setLevel(self._log_level)
    else:
      logging.disable()

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn" and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(min(self._log_level, logging.INFO))
      logger.addHandler(queue_handler)

    self._logger = logger
