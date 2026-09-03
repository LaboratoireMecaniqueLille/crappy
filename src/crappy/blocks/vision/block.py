# coding: utf-8

from abc import ABC
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import (synchronize, managers, RLock, Event, Value,
                             sharedctypes)
import numpy as np
import logging
from typing import Any
from dataclasses import dataclass, field
from uuid import uuid4
from math import prod

from ..meta_block import Block
from ...links import ImageLink
from ..._global import LinkDataError, PrepareError


@dataclass
class ImgLinkData:
  """Class containing all the attributes that are needed for sending or
  receiving images through :class:`~crappy.links.ImageLink`.

  Useful as a convenience to enforce a clear data structure.

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
  """Class containing all the information about the last received image from an
  upstream :class:`~crappy.links.ImageLink`.

  .. versionadded:: 2.1.0
  """

  id: int = -1
  metadata: dict[str, Any] | None = None
  img: np.ndarray = field(default_factory=lambda: np.empty(0))


class VisionBlock(Block, ABC):
  """Base class for Blocks that can send and/or receive images in Crappy.
  
  It implements all the mechanisms necessary for handling images, such as the
  generation, sharing and cleanup of synchronization objects and shared 
  buffers, or methods for sending and receiving images.
  
  This class cannot be used as-is, since it doesn't actually perform any 
  action. It simply exposes methods for other Blocks to use.
  
  It works in combination with :class:`~crappy.links.ImageLink`, through which
  images are sent and received.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               img_shape: tuple[int, int] | tuple[int, int, int] | None = None,
               img_dtype: str | None = None,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 200) -> None:
    """Sets the arguments and initializes the attributes.
    
    Args:
      img_shape: The shape of the images that this Block sends to downstream 
        Blocks. Set to :obj:`None` if it doesn't output images. The shape 
        should be given as a :obj:`tuple` of :obj:`int`, as returned by 
        :obj:`numpy.shape`. **This argument is mandatory in case the Block 
        doesn't have a configuration window/mechanism.** If a configuration is
        used, the value of this argument is ignored.
      img_dtype: The dtype of the images that this Block sends to downstream 
        Blocks. Set to :obj:`None` if it doesn't output images. The dtype 
        should be given as a :obj:`str`, as returned by :obj:`numpy.dtype`. 
        **This argument is mandatory in case the Block doesn't have a
        configuration window/mechanism.** If a configuration is used, the value 
        of this argument is ignored.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
    """

    super().__init__()

    # Indicates that this Block is meant for handling images
    self.is_vision_block = True

    # Validate Block-level arguments before setting them
    if (freq is not None and
        (not (isinstance(freq, int) or isinstance(freq, float)) or
         freq <= 0)):
      raise ValueError("If provided, freq should be a strictly positive float")
    if debug is not None and not isinstance(debug, bool):
      raise ValueError("If provided, debug should be a boolean")
    if display_freq is not None and not isinstance(display_freq, bool):
      raise ValueError("If provided, display_freq should be a boolean")
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # The lists of input and output ImageLinks
    self.img_outputs: list[ImageLink] = list()
    self.img_inputs: list[ImageLink] = list()

    # If provided, the shape and dtype must be valid
    if (img_shape is not None and
       (not isinstance(img_shape, tuple) or
        len(img_shape) not in (2, 3) or
        not all(isinstance(el, int) for el in img_shape))):
      raise ValueError("The image shape should be a 2- or 3-tuple of "
                       "integers")
    if img_dtype is not None and not isinstance(img_dtype, str):
      raise ValueError("The image dtype should be a valid Numpy type, "
                       "provided as a string")

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

  def prepare(self) -> None:
    """Retrieves the shared buffers and shared arrays from upstream 
    :class:`~crappy.links.ImageLink`, then sets the shared array and makes it 
    available to downstream :class:`~crappy.links.ImageLink`."""

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

  def finish(self) -> None:
    """Ensures that the resources from the SharedMemory are released."""

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
    """Adds an output :class:`~crappy.links.ImageLink` to the list of output
    ImageLinks of the Block."""

    self.img_outputs.append(img_link)

  def add_img_input(self, img_link: ImageLink) -> None:
    """Adds an input :class:`~crappy.links.ImageLink` to the list of input
    ImageLinks of the Block"""

    self.img_inputs.append(img_link)

    # Create the buffer for the last received image
    self.last_received[img_link.name] = ImgData()

  def send_img(self, metadata: dict[str, Any], img: np.ndarray) -> None:
    """Sends an image to downstream Blocks.

    In practice, the image is written in a share memory object accessible by
    all the downstream Blocks. The metadata associated to the frame is written
    separately in a shared dictionary.

    Args:
      metadata: A :obj:`dict` containing the metadata associated to the image.
      img: The image to send, as a :obj:`numpy.ndarray`.
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
      raise ValueError("The metadata to send must constain an 'ImageUniqueID' "
                       "key")
    if 't(s)' not in metadata:
      raise ValueError("The metadata to send must constain a 't(s)' key")

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
    """Checks incoming :class:`~crappy.links.ImageLink` for new images, and
    copies the new images and their metadata to a local buffer.

    The ImageLinks with no new image are simply ignored.

    Returns:
      A :obj:`list` containing the names of all the ImageLinks whose images
      were grabbed.
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

  def set_shared_objects(self) -> None:
    """Sets the shared objects required for sending images, and shares them
    with all the :class:`~crappy.links.ImageLink`.

    Called by the master Block in the `__main__` Process, before this Block's
    Process even starts.
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
    """Retrieves the shared objects necessary for receiving images from the
    incoming ImageLinks."""

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
    """Initializes the image buffer and metadata buffer for sending images to
    downstream Blocks.

    Also raises a flag indicating downstream Blocks that the buffers are ready
    to use.

    Args:
      img_shape: The shape of the images to share, as a :obj:`tuple` of
        `obj:`int`, as returned by ``array.shape``.
      dtype: The dtype of the image data, as a :obj:`str`, and as returned by
        a call to ``array.dtype``.
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
    """Retrieves the image buffer and its associated Numpy array for receiving
    images from an upstream Block.

    Args:
      name: The name of the Shared Memory object that will contain the shared
        images, as a :obj:`str`.
      buffer_ready: The Event indicating when the shared buffer is ready to be
        used.
      img_info_dict: The shared dictionary containing the image shape and
        dtype.

    Returns:
      The shared memory and the convenience Numpy buffer obtained using the
      provided name.
    """

    if self._ready_barrier is None:
      raise ValueError("The ready Barrier should be set at this point")

    # Periodically checks if Crappy has crashed, otherwise waits for the
    # upstream buffer to be available
    while not buffer_ready.wait(0.5):
      self.log(logging.DEBUG, f"Buffers with name {name} not ready yet")
      if self._ready_barrier.broken:
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
