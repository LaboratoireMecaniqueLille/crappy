# coding: utf-8

from abc import ABC
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import (synchronize, managers, RLock, Event, Value,
                             sharedctypes, connection)
import numpy as np
import logging
from typing import Any
from dataclasses import dataclass, field
from uuid import uuid4
from math import prod
from time import time

from ..meta_block import Block
from ...links import ImageLink
from ..._global import LinkDataError, PrepareError
from ...tool.camera_config import CameraConfig


@dataclass
class ConfigRequest:
  """Description of a configuration requested from an image source.

  The request is created by a downstream :class:`VisionBlock` in
  :meth:`VisionBlock.request_config`. Before the Blocks start, Crappy
  duplicates it and assigns one end of a one-way :obj:`multiprocessing.Pipe` to
  the source and the other end to the requester.

  Args:
    requester: Name of the downstream Block requesting configuration.
    args: Positional arguments forwarded to the requested configurator.
    kwargs: Keyword arguments forwarded to the requested configurator.
    configurator: :class:`~crappy.tool.camera_config.CameraConfig` subclass to
      instantiate on the image source.
    img_source: Name of the upstream image source handling the request.
    connection: Pipe endpoint assigned by
      :meth:`~crappy.blocks.Block.prepare_all`. Requesters receive a readable
      endpoint and sources receive a writable one.
    completed: Whether the source successfully sent a response.
    required: Whether the requester can start without configuration data. A
      required request must receive a non-:obj:`None` response.

  .. versionadded:: 2.1.0
  """

  requester: str
  args: tuple[Any, ...]
  kwargs: dict[str, Any]
  configurator: type[CameraConfig]
  img_source: str
  connection: connection.Connection | None = None
  completed: bool = False
  required: bool = True

  def __post_init__(self) -> None:
    """Validates the types of the mutable fields."""

    if not isinstance(self.completed, bool):
      raise TypeError("completed must be a boolean")
    if not isinstance(self.required, bool):
      raise TypeError("required must be a boolean")


@dataclass
class ImgLinkData:
  """Groups the shared-memory state associated with one ImageLink endpoint.

  Args:
    memory_name: Name of the shared-memory segment containing image data.
    img_lock: Process-safe lock guarding a consistent image and metadata read
      or write.
    metadata_dict: Shared dictionary containing the current image metadata.
    buffer_ready: Event set after the image buffer has been created.
    img_info_dict: Shared dictionary containing the image shape and dtype.
    img_id: Shared counter identifying the current buffer contents.
    img_buffer: Shared-memory handle owned or attached by this Block.
    npy_buffer: :mod:`numpy` view over ``img_buffer``.

  .. versionadded:: 2.1.0
  """

  # Common to inputs and outputs, always set or get together
  memory_name: str | None = None
  img_lock: synchronize.RLock | None = None
  metadata_dict: managers.DictProxy | None = None
  buffer_ready: synchronize.Event | None = None
  img_info_dict: managers.DictProxy | None = None
  img_id: sharedctypes.Synchronized | None = None

  # Available for both inputs and outputs, but not set at the same moment
  img_buffer: SharedMemory | None = None
  npy_buffer: np.ndarray | None = None


@dataclass
class ImgData:
  """Stores the latest image copied from one upstream ImageLink.

  Args:
    id: Last transport-level image identifier handled by this Block.
    metadata: Metadata copied together with the image.
    img: Local :mod:`numpy` buffer containing the copied image.

  .. versionadded:: 2.1.0
  """

  id: int = -1
  metadata: dict[str, Any] | None = None
  img: np.ndarray = field(default_factory=lambda: np.empty(0))


class VisionBlock(Block, ABC):
  """Base class for Blocks that exchange images through shared memory.

  :class:`VisionBlock` extends :class:`~crappy.blocks.Block` with input and
  output :class:`~crappy.links.ImageLink` support. Regular Links remain
  available for commands, measurements, metadata, and overlays, while
  ImageLinks carry image arrays and their metadata without serializing the
  image through a Pipe.

  Each image source owns one shared-memory buffer for all of its downstream
  ImageLinks. :meth:`send_img` updates that buffer atomically under a shared
  lock, and :meth:`receive_imgs` copies its newest contents into a local buffer
  in each consumer. ImageLinks therefore expose the latest frame rather than a
  queue: a slow consumer can skip intermediate images, but never reads a
  partially updated image or mismatched metadata.

  This class also implements source-side configuration requests. Before Block
  processes start, a downstream VisionBlock can return a
  :class:`ConfigRequest` from :meth:`request_config`. Crappy routes that
  request to the relevant upstream image source and creates a one-way Pipe for
  the response. Sources answer with :meth:`send_config`, while requesters
  collect responses with :meth:`recv_configs` during preparation.

  Subclasses define their supported ImageLink topology and implement the
  actual acquisition, processing, display, or recording behavior. This class
  manages buffer creation, attachment, synchronization, and cleanup.

  Unlike :class:`~crappy.blocks.camera_processes.CameraProcess`, which is a
  helper Process owned by the older :class:`~crappy.blocks.Camera`, a
  VisionBlock is a complete :class:`~crappy.blocks.Block` with its own graph
  node, regular Links, lifecycle, logging, and loop-frequency control.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               img_shape: tuple[int, int] | tuple[int, int, int] | None = None,
               img_dtype: str | None = None,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 200) -> None:
    """Sets the output image format and standard Block options.

    Args:
      img_shape: Shape of images sent through output ImageLinks, as a two- or
        three-item tuple matching :attr:`numpy.ndarray.shape`. It can initially
        be :obj:`None` when a subclass determines the format during
        configuration, and is unnecessary for Blocks without output
        ImageLinks. Otherwise, it must be known before :meth:`prepare` creates
        the shared buffer.
      img_dtype: Dtype of images sent through output ImageLinks, written as a
        non-empty string accepted by :func:`numpy.dtype`. Like ``img_shape``,
        it can initially be :obj:`None` when discovered during configuration or
        when the Block has no output ImageLinks.
      display_freq: If :obj:`True`, periodically reports the rate at which
        images are actually handled.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: Target looping frequency for this Block. If :obj:`None`, loops as
        fast as possible. This limits how often a subclass can check or handle
        images, not the frequency of an upstream source.
    """

    super().__init__()

    # Set Block-level arguments
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug
    self.is_vision_block = True

    # The lists of input and output ImageLinks
    self.img_outputs: list[ImageLink] = list()
    self.img_inputs: list[ImageLink] = list()

    # List of configuration requests received from downstream Blocks
    # Access it through the property, not this private attribute
    self._config_requests_in: list[ConfigRequest] = list()
    # List of configuration requests emitted by this Block
    self._config_requests_out: list[ConfigRequest] = list()

    # If provided, the shape and dtype must be valid
    if img_shape is not None and not isinstance(img_shape, tuple):
      raise TypeError("When provided, img_shape must be a tuple of 2 or 3 "
                      "strictly positive integers")
    if img_shape is not None and not 1 < len(img_shape) < 4:
      raise ValueError("When provided, img_shape must be a tuple of 2 or 3 "
                       "strictly positive integers")
    if (img_shape is not None and
        not all(isinstance(el, int) for el in img_shape)):
      raise ValueError("When provided, img_shape must be a tuple of 2 or 3 "
                       "strictly positive integers")
    if (img_dtype is not None and
        (not isinstance(img_dtype, str) or not img_dtype)):
      raise ValueError("When provided, img_dtype must be a non-empty string")

    # Information on the output images
    self._img_shape: tuple[int, int] | tuple[int, int, int] | None = img_shape
    self._img_dtype: str | None = img_dtype

    # The list of ImgLinkData objects corresponding to the input ImageLinks
    self._in_link_data: list[ImgLinkData] = list()
    # The objects containing for each ImageLink the last received image
    self.last_received: dict[str, ImgData] = dict()

    # Objects for sharing images with downstream Blocks if needed
    self._out_link_data = ImgLinkData()

    # Counter keeping track of the number of images that were sent
    self._sent_img_counter: int = 0

    # Attributes for displaying the FPS counter
    self._loop_count = 0
    self._fps_count = 0
    self._last_fps_img = time()

  def prepare(self) -> None:
    """Creates outgoing image buffers and attaches to incoming ones.

    Any configuration requests received by this Block as an image source must
    already have been answered. The method then retrieves synchronization
    objects from all input ImageLinks, creates one shared output buffer when
    needed, waits for upstream buffers to become available, and creates a local
    receive buffer for each input.

    Subclasses overriding this method normally validate their ImageLink
    topology and complete format/configuration setup before calling
    ``super().prepare()``.

    Raises:
      RuntimeError: If a configuration request was not handled or an incoming
        ImageLink has no shared-buffer information.
      ValueError: If output shape or dtype information is missing or invalid,
        or if shared synchronization and buffer objects are inconsistent.
      PrepareError: If another Block fails while this Block waits for an
        upstream image buffer.
    """

    # If at that point not all requests have been handled, they will never be
    if not all(request.completed for request in self.config_requests_in):
      unhandled = [request.requester for request in self.config_requests_in
                   if not request.completed]
      raise RuntimeError(f"Not all configuration requests were handled, "
                         f"aborting!\nNo configuration was provided to Blocks"
                         f"{', '.join(unhandled)}\nTry to adjust this Block's "
                         f"arguments, or provide configuration information to "
                         f"the downstream Blocks")

    # First getting the synchronization objects, but not yet the image buffers
    # This call should return almost immediately
    self._get_shared_objects()

    # Only set downstream image buffers if there are output ImageLinks
    if self.img_outputs:
      # The image shape and dtype must be known upfront and valid
      if self._img_shape is None or self._img_dtype is None:
        raise ValueError("The image shape and dtype weren't provided, cannot "
                         "initialize the downstream image buffers")
      if (not isinstance(self._img_shape, tuple) or
          len(self._img_shape) not in (2, 3) or
          not all(isinstance(el, int) for el in self._img_shape)):
        raise ValueError("The image shape should be a 2- or 3-tuple of "
                         "integers")
      if not isinstance(self._img_dtype, str):
        raise ValueError("The image dtype should be a valid Numpy type, "
                         "provided as a string")

      # Setting the downstream image buffers
      self._set_image_buffer(self._img_shape, self._img_dtype)

    # Getting the actual image buffer for each incoming ImageLink
    for link, data in zip(self.img_inputs, self._in_link_data):
      # Perform checks to avoid unexpected cases
      if data.memory_name is None:
        raise ValueError("Cannot get shared buffer as its name was not set")
      if data.buffer_ready is None:
        raise ValueError("Cannot get shared buffer as its shared Event was "
                         "not set")
      if data.img_info_dict is None:
        raise ValueError("Cannot get shared buffer as its shared information "
                         "dictionary was not set")

      # Should block until CameraConfig exits, or Crappy crashes
      (data.img_buffer,
       data.npy_buffer) = self._get_image_buffer(data.memory_name,
                                                 data.buffer_ready,
                                                 data.img_info_dict)
      self.log(logging.INFO, f"Received shared image buffers from ImageLink "
                             f"{link.name}")

      if data.npy_buffer is None:
        raise ValueError("The Numpy array buffer was not set")

      # Also initialize the last received image buffer
      self.last_received[link.name].img = np.empty(
          shape=data.npy_buffer.shape, dtype=data.npy_buffer.dtype)

  def begin(self) -> None:
    """Starts handled-image frequency measurement when the test begins."""

    self._last_fps_img = time()

  def finish(self) -> None:
    """Releases shared-memory resources owned or attached by this Block.

    Incoming shared-memory handles are closed without unlinking their
    source-owned segments. The output segment, when present, is both closed and
    unlinked by its owning Block.
    """

    # Close the SharedMemory objects of incoming ImageLinks
    if hasattr(self, '_in_link_data'):
      for data in self._in_link_data:
        if data.img_buffer is not None:
          data.img_buffer.close()
      self.log(logging.INFO, "Closed shared image buffers from upstream "
                             "Blocks")

    # Same for the downstream ImageLinks, except we also have to unlink()
    if hasattr(self, '_out_link_data'):
      if self._out_link_data.img_buffer is not None:
        self._out_link_data.img_buffer.close()
        self._out_link_data.img_buffer.unlink()
        self.log(logging.INFO, "Closed image buffer shared with downstream "
                               "Blocks")

  def add_img_output(self, img_link) -> None:
    """Registers an ImageLink through which this Block sends images.

    Args:
      img_link: Output :class:`~crappy.links.ImageLink` being connected.
    """

    self.img_outputs.append(img_link)

  def add_img_input(self, img_link: ImageLink) -> None:
    """Registers an ImageLink from which this Block receives images.

    A placeholder :class:`ImgData` entry is created immediately and populated
    with a correctly shaped local buffer during :meth:`prepare`.

    Args:
      img_link: Input :class:`~crappy.links.ImageLink` being connected.
    """

    self.img_inputs.append(img_link)

    # Create the buffer for the last received image
    self.last_received[img_link.name] = ImgData()

  def send_img(self, metadata: dict[str, Any], img: np.ndarray) -> None:
    """Publishes an image and its metadata to all output ImageLinks.

    Under the output lock, the metadata proxy and shared :mod:`numpy` buffer
    are updated before the transport-level image counter is incremented.
    Consumers use that counter to determine whether a new frame is available.
    The supplied metadata must contain ``'t(s)'`` and ``'ImageUniqueID'``.
    The image shape and dtype must exactly match the prepared shared buffer.

    Args:
      metadata: Metadata associated with the image.
      img: Image array to copy into shared memory.

    Raises:
      LinkDataError: If ``metadata`` is not a dictionary or ``img`` is not a
        :class:`numpy.ndarray`.
      ValueError: If shared objects are unavailable, mandatory metadata keys
        are missing, or the image format differs from the prepared buffer.
    """

    # Checking data integrity before sending
    if not isinstance(metadata, dict):
      self.log(logging.ERROR, f"Trying to send metadata of type "
                              f"{type(metadata).__name__} instead of dict!")
      raise LinkDataError
    if not isinstance(img, np.ndarray):
      self.log(logging.ERROR, f"Trying to send image of type "
                              f"{type(img).__name__} instead of Numpy array!")
      raise LinkDataError

    # Checking shared object availability before sending
    if self._out_link_data.img_lock is None:
      raise ValueError("Cannot send image because the image lock isn't "
                       "initialized")
    if self._out_link_data.npy_buffer is None:
      raise ValueError("Cannot send image because the image buffer isn't "
                       "initialized")
    if self._out_link_data.metadata_dict is None:
      raise ValueError("Cannot send image because the shared metadata "
                       "dictionary isn't initialized")
    if self._out_link_data.img_id is None:
      raise ValueError("Cannot send image because the shared image ID counter "
                       "isn't initialized")

    # Make sure the mandatory keys are provided
    if 'ImageUniqueID' not in metadata:
      raise ValueError("The metadata to send must contain an 'ImageUniqueID' "
                       "key")
    if 't(s)' not in metadata:
      raise ValueError("The metadata to send must contain a 't(s)' key")

    # Double-check image type and dtype consistency
    if img.dtype != self._out_link_data.npy_buffer.dtype:
      raise ValueError(f"The dtype of the image to send ({img.dtype}) "
                       f"doesn't match the one of the image buffer "
                       f"({self._out_link_data.npy_buffer.dtype})")
    if img.shape != self._out_link_data.npy_buffer.shape:
      raise ValueError(f"The shape of the image to send ({img.shape}) "
                       f"doesn't match the one of the image buffer "
                       f"({self._out_link_data.npy_buffer.shape})")

    with self._out_link_data.img_lock:
      # Sending the metadata dictionary
      self.log(logging.DEBUG, f"Writing metadata to shared dict: {metadata}")
      self._out_link_data.metadata_dict.clear()
      self._out_link_data.metadata_dict.update(metadata)
      # Sending the actual image
      self.log(logging.DEBUG, "Writing image to shared memory")
      np.copyto(self._out_link_data.npy_buffer, img)
      # Sending the unique image ID
      self.log(logging.DEBUG, "Updating image ID")
      self._out_link_data.img_id.value = self._sent_img_counter
      self._sent_img_counter += 1

  def receive_imgs(self) -> list[str]:
    """Copies the newest frame available on each input ImageLink.

    Each source is checked under its shared lock. If its transport-level image
    identifier differs from the last handled identifier, both metadata and
    image data are copied into :attr:`last_received`. Sources without a new
    frame are ignored. Since an ImageLink owns only one shared buffer,
    intermediate frames may have been overwritten by the newest one.

    Returns:
      Names of the ImageLinks from which a new image was copied.

    Raises:
      ValueError: If shared synchronization objects or buffers are unavailable,
        or if local and shared image formats are inconsistent.
    """

    updated: list[str] = list()

    # Iterate over each incoming ImageLink
    for link, data_in in zip(self.img_inputs, self._in_link_data):

      # Fail early in case of inconsistent state
      if data_in.img_lock is None:
        raise ValueError("No image Lock set for this ImageLink!")

      # Guard reading and writing against race conditions
      with data_in.img_lock:

        # Fail early in case of inconsistent state
        if data_in.img_id is None:
          raise ValueError("No image ID counter set for this ImageLink!")

        # If the image ID is the same as the stored one, there's no new image
        # to receive
        if data_in.img_id.value == self.last_received[link.name].id:
          self.log(logging.DEBUG, f"No new image to grab for link {link.name}")
          continue

        # Checking shared object integrity before reading data
        if data_in.metadata_dict is None:
          raise ValueError("No metadata dictionary set for this ImageLink!")
        if not self.last_received[link.name].img.nbytes:
          raise ValueError("The received image buffer was not initialized")
        if data_in.npy_buffer is None:
          raise ValueError("No shared Numpy buffer set for this ImageLink!")

        # Double-check image type and dtype consistency
        if (data_in.npy_buffer.dtype !=
            self.last_received[link.name].img.dtype):
          raise ValueError(f"The dtype of the image to send "
                           f"({data_in.npy_buffer.dtype}) "
                           f"doesn't match the one of the image buffer "
                           f"({self.last_received[link.name].img.dtype})")
        if (data_in.npy_buffer.shape !=
            self.last_received[link.name].img.shape):
          raise ValueError(f"The shape of the image to send "
                           f"({data_in.npy_buffer.shape}) "
                           f"doesn't match the one of the image buffer "
                           f"({self.last_received[link.name].img.shape})")

        # Retrieve all the information from the input buffer
        self.last_received[link.name].metadata = data_in.metadata_dict.copy()
        self.log(logging.DEBUG, f"Received metadata dict from link "
                                f"{link.name}")
        np.copyto(self.last_received[link.name].img, data_in.npy_buffer)
        self.log(logging.DEBUG, f"Received image from link {link.name}")
        self.last_received[link.name].id = data_in.img_id.value
        self.log(logging.DEBUG, f"Received unique ID from link {link.name}")

      # Store to indicate that the data from this ImageLink was updated
      updated.append(link.name)

    self.log(logging.DEBUG, f"Data received during this call from ImageLinks: "
                            f"{', '.join(updated)}")
    return updated

  def request_config(self, source: str) -> ConfigRequest | None:
    """Returns this Block's configuration request for an image source.

    Subclasses requiring source-side interactive configuration should override
    this method and return a :class:`ConfigRequest`. The default implementation
    makes no request.

    Args:
      source: Name of an upstream image source.

    Returns:
      The configuration request for that source, or :obj:`None` when no
      configuration is required.
    """

    return None

  def add_config_request_in(self, request: ConfigRequest) -> None:
    """Registers a configuration request received from a downstream Block.

    Args:
      request: Request containing the writable Pipe endpoint assigned to this
        source.

    Raises:
      TypeError: If *request* is not a :class:`ConfigRequest`.
      RuntimeError: If no Pipe endpoint was assigned to the request.
    """

    if not isinstance(request, ConfigRequest):
      raise TypeError("The request should be a ConfigRequest")
    if request.connection is None:
      raise RuntimeError("The ConfigRequest is expected to have its "
                         "Connection set")
    self._config_requests_in.append(request)

  @property
  def config_requests_in(self) -> list[ConfigRequest]:
    """Configuration requests this image source received from consumers."""

    return self._config_requests_in

  def send_config(self,
                  request: ConfigRequest,
                  config: tuple[Any, ...] | None) -> None:
    """Sends a configuration response to a downstream Block.

    The request is marked as completed after a successful send. Its Pipe
    endpoint is closed whether sending succeeds or fails.

    Args:
      request: Incoming request to answer.
      config: Configuration data to send, or :obj:`None` when an optional
        request was declined or no configuration can be provided.

    Raises:
      RuntimeError: If the request has no Pipe endpoint or a required request
        is answered with :obj:`None`.
    """

    try:
      if request.connection is None:
        raise RuntimeError("The ConfigRequest is expected to have its "
                           "Connection set")
      # Checking if a required request was properly handled
      if request.required and config is None:
        raise RuntimeError(f"The request from Block {request.requester} is "
                           f"flagged as required, yet {request.img_source} "
                           f"answered with None")

      request.connection.send(config)
      self.log(logging.INFO, f"Sent "
                             f"{'config' if config is not None else 'None'} "
                             f"back to Block {request.requester}")
      request.completed = True
    except (Exception,):
      self.log(logging.ERROR, "Couldn't send the config info via the "
                              "Connection!")
      raise
    finally:
      if request.connection is not None:
        request.connection.close()

  def add_config_request_out(self, request: ConfigRequest) -> None:
    """Registers a configuration request sent to an upstream image source.

    Args:
      request: Request containing the readable Pipe endpoint assigned to this
        consumer.

    Raises:
      TypeError: If *request* is not a :class:`ConfigRequest`.
      RuntimeError: If no Pipe endpoint was assigned to the request.
    """

    if not isinstance(request, ConfigRequest):
      raise TypeError("The request should be a ConfigRequest")
    if request.connection is None:
      raise RuntimeError("The ConfigRequest is expected to have its "
                         "Connection set")
    self._config_requests_out.append(request)

  def recv_configs(self) -> dict[str, tuple[Any, ...] | None]:
    """Waits for all requested source configurations and returns them.

    While waiting, this method periodically checks whether another Block broke
    the preparation barrier or requested the test to stop. Each readable Pipe
    endpoint is closed after receiving its one response or encountering an
    error.

    Returns:
      A dictionary associating image-source names with their configuration
      data.

    Raises:
      ValueError: If the preparation synchronization objects are unavailable.
      RuntimeError: If a request has no Pipe endpoint.
      PrepareError: If another Block fails, the test stops, or the source Pipe
        closes without providing a response.
    """

    if self._ready_barrier is None:
      raise ValueError("The ready Barrier should be set at this point")
    if self._stop_event is None:
      raise ValueError("The stop Event should be initialized at this point")

    configs: dict[str, tuple[Any, ...] | None] = dict()

    # For each request, wait for the configuration information to arrive
    for request in self._config_requests_out:
      if request.connection is None:
        raise RuntimeError("The ConfigRequest is expected to have a "
                           "Connection set")
      try:
        # Wait until config information is received
        while not request.connection.poll(timeout=0.5):
          self.log(logging.DEBUG, f"Config from Block {request.img_source} "
                                  f"not ready yet")
          # Check if we should give up on waiting for the config
          if self._ready_barrier.broken or self._stop_event.is_set():
            raise PrepareError("An exception occurred in another Block, "
                               "aborting")

        # Receive the configuration information
        config = request.connection.recv()
        self.log(logging.DEBUG, f"Received config information from Block "
                                f"{request.img_source}")
      # Can happen if the other end of the Pipe is broken
      except (EOFError, OSError):
        raise PrepareError("The other end of the Pipe seems to be broken")

      finally:
        request.connection.close()

      # Just store the received configuration
      configs[request.img_source] = config

    return configs

  def set_shared_objects(self) -> None:
    """Creates and distributes synchronization objects for output images.

    This method runs in the main Process before Block processes start. When the
    Block has output ImageLinks, it creates the shared-memory name, lock,
    metadata and format proxies, readiness event, and image identifier used by
    every downstream consumer. The objects are then registered on each output
    ImageLink. The actual shared-memory segment is created later in
    :meth:`prepare`, once the output format is final.

    Raises:
      ValueError: If the shared Manager is unavailable or synchronization
        objects could not be initialized completely.
    """

    # If there's no downstream image Block, no need for synchronization objects
    if not self.img_outputs:
      return

    self._out_link_data.memory_name = f"{self.name}_img_buffer_{uuid4()}"
    self._out_link_data.img_lock = RLock()
    self._out_link_data.buffer_ready = Event()
    if self.shared_mgr is not None:
      self._out_link_data.metadata_dict = (self.shared_mgr.dict())
      self._out_link_data.img_info_dict = (self.shared_mgr.dict())
    else:
      raise ValueError("The base Manager hasn't been initialized yet!")
    self._out_link_data.img_id = Value('l')
    self._out_link_data.img_id.value = -1

    # Share the buffer objects with the provided downstream ImageLinks
    for img_link in self.img_outputs:
      if (self._out_link_data.memory_name is not None
          and self._out_link_data.img_lock is not None
          and self._out_link_data.metadata_dict is not None
          and self._out_link_data.buffer_ready is not None
          and self._out_link_data.img_info_dict is not None
          and self._out_link_data.img_id is not None):
        img_link.set_buffers(self._out_link_data.memory_name,
                             self._out_link_data.img_lock,
                             self._out_link_data.metadata_dict,
                             self._out_link_data.buffer_ready,
                             self._out_link_data.img_info_dict,
                             self._out_link_data.img_id)
      else:
        raise ValueError("Not all synchronization objects have been "
                         "initialized yet")

  def _get_shared_objects(self) -> None:
    """Retrieves synchronization objects from all input ImageLinks.

    Raises:
      RuntimeError: If an ImageLink does not expose complete shared-buffer
        information.
    """

    for link in self.img_inputs:
      if (ret := link.get_buffers()) is not None:
        self._in_link_data.append(ImgLinkData(*ret))
        self.log(logging.DEBUG, f"Got image buffer objects from link "
                                f"{link.name}")
      else:
        raise RuntimeError(f"The ImageLink {link.name} does not contain "
                           f"shared buffer data!")

  def _set_image_buffer(self,
                        img_shape: tuple[int, int] | tuple[int, int, int],
                        dtype: str) -> None:
    """Creates the shared image array used by all output ImageLinks.

    After allocating the named shared-memory segment, this method creates a
    :mod:`numpy` view over it, publishes its shape and dtype, and sets the
    readiness event so downstream Blocks can attach.

    Args:
      img_shape: Shape of the images to share, as returned by
        :attr:`numpy.ndarray.shape`.
      dtype: Image dtype as a string accepted by :func:`numpy.dtype`.

    Raises:
      ValueError: If the shared-memory name, information proxy, or readiness
        event has not been initialized.
    """

    if self._out_link_data.memory_name is None:
      raise ValueError("Cannot initialize the shared memory as its name was "
                       "never set")

    # First, set the shared memory containing image data
    self._out_link_data.img_buffer = SharedMemory(
        name=self._out_link_data.memory_name,
        create=True,
        size=prod(img_shape) * np.dtype(dtype).itemsize)
    self.log(logging.DEBUG, "Initialized the SharedMemory object")

    if self._out_link_data.img_buffer is None:
      raise ValueError("Cannot initialize the shared array if the shared "
                       "memory is None")

    # For convenience, use a Numpy array as a proxy to the shared memory
    self._out_link_data.npy_buffer = np.ndarray(
        img_shape,
        dtype=np.dtype(dtype),
        buffer=self._out_link_data.img_buffer.buf)
    self.log(logging.DEBUG, "Initialized the Numpy array for sharing images")

    if self._out_link_data.img_info_dict is None:
      raise ValueError("Cannot share image shape and dtype if the shared "
                       "dictionary is None")

    # Share the dtype and shape of the image
    self._out_link_data.img_info_dict.update({'shape': img_shape,
                                              'dtype': dtype})
    self.log(logging.DEBUG, "Shared image shape and dtype with downstream "
                            "Blocks")

    if self._out_link_data.buffer_ready is None:
      raise ValueError("Cannot set the buffer ready event if it is None")

    # Set the event indicating that the shared memory is ready
    self._out_link_data.buffer_ready.set()
    self.log(logging.DEBUG, "Set the buffer_ready Event")

  def _get_image_buffer(self,
                        name: str,
                        buffer_ready: synchronize.Event,
                        img_info_dict: managers.DictProxy
                        ) -> tuple[SharedMemory, np.ndarray]:
    """Attaches to an upstream shared image buffer.

    The method waits for the source to publish its image shape and dtype, while
    periodically checking whether preparation was aborted. It then opens the
    named shared-memory segment and creates a :mod:`numpy` view over it.

    Args:
      name: Name of the upstream shared-memory segment.
      buffer_ready: Event indicating that the source buffer is ready.
      img_info_dict: Shared dictionary containing image ``'shape'`` and
        ``'dtype'`` entries.

    Returns:
      The attached shared-memory handle and its :mod:`numpy` array view.

    Raises:
      ValueError: If preparation synchronization objects or image format
        entries are unavailable.
      PrepareError: If another Block fails or the test stops while waiting for
        the source buffer.
    """

    if self._ready_barrier is None:
      raise ValueError("The ready Barrier should be set at this point")
    if self._stop_event is None:
      raise ValueError("The stop Event should be initialized at this point")

    # Periodically checks if Crappy has crashed, otherwise waits for the
    # upstream buffer to be available
    while not buffer_ready.wait(0.5):
      self.log(logging.DEBUG, f"Buffers with name {name} not ready yet")
      if self._ready_barrier.broken or self._stop_event.is_set():
        raise PrepareError("An exception occurred in another Block, aborting")
    self.log(logging.DEBUG, f"Buffer with name {name} ready to be shared")

    if 'shape' not in img_info_dict or 'dtype' not in img_info_dict:
      raise ValueError("The shared dict containing image information should "
                       "expose the 'shape' and 'dtype' keys")

    # Retrieve the shape and dtype from shared information
    shape, dtype = img_info_dict['shape'], img_info_dict['dtype']

    # Instantiate the shared memory and the convenience Numpy array buffers
    img_buffer = SharedMemory(name=name, create=False)
    npy_buffer = np.ndarray(shape,
                            dtype=np.dtype(dtype),
                            buffer=img_buffer.buf)

    return img_buffer, npy_buffer

  def _print_freq(self, img_handled: bool) -> None:
    """Periodically logs the achieved handled-image frequency.

    The count can differ from the Block's loop frequency because loops that do
    not acquire, process, display, or save an image are excluded.

    Args:
      img_handled: Whether an image was handled during the current loop.
    """

    self._fps_count += int(img_handled)
    t = time()
    if t - self._last_fps_img > 2:
      self.log(logging.INFO, f"Frames handled per second: "
                             f"{self._fps_count / (t - self._last_fps_img)}")
      self._last_fps_img = t
      self._fps_count = 0
