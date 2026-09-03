# coding: utf-8

from multiprocessing import (synchronize, current_process, managers,
                             sharedctypes)
import logging


class ImageLink:
  """This class is used for transferring images between two instances of
  :class:`~crappy.blocks.Block`.

  The created ImageLink is unidirectional, from the input Block to the output
  Block. Under the hood, an ImageLink mostly manages the exchange of
  information like synchronization objects and image buffers between the input
  and the output Blocks.

  This class should not be mistaken with :class:`~crappy.links.Link`, that can
  only transfer dictionaries between Blocks, not images.

  .. versionadded:: 2.1.0
  """

  names: list[str] = list()
  _count: int = 0

  def __init__(self,
               input_block,
               output_block,
               name: str | None = None) -> None:
    """Sets the instance attributes.

    Args:
      input_block: The Block sending images through the ImageLink.
      output_block: The Block receiving images through the ImageLink.
      name: Name of the ImageLink, to differentiate it from the others. If no
        specific name is given, the ImageLinks are numbered in the
        order in which they are instantiated in the script.
    """

    self.name = self._get_name(name)
    self._img_buffer_name: str | None = None
    self._img_lock: synchronize.RLock | None = None
    self._metadata_proxy: managers.DictProxy | None = None
    self._buffer_ready: synchronize.Event | None = None
    self._img_info_proxy: managers.DictProxy | None = None
    self._img_id: sharedctypes.Synchronized | None = None

    # Associating the img_link to the input and output Blocks
    input_block.add_img_output(self)
    output_block.add_img_input(self)

    self._logger: logging.Logger | None = None

  def __new__(cls, *args, **kwargs):
    """When instantiating a new ImageLink, increments the ImageLink counter."""

    cls._count += 1
    return super().__new__(cls)

  @classmethod
  def _get_name(cls, name: str | None) -> str:
    """Returns a suitable name for this ImageLink, or checks the validity of
    the one provided if any.

    Args:
      name: The provided name for the ImageLink, either :obj:`None` or a
        :obj:`str`.

    Returns:
      The name generated or validated for this ImageLink, as a :obj:`str`.
    """

    # If the provided name is unique, OK
    if name is not None and name not in cls.names:
      cls.names.append(name)
      return name
    # If the provided name is a duplicate, raise
    elif name is not None and name in cls.names:
      raise ValueError(f"Name {name} already given to a different ImageLink, "
                       f"please use a different name!")
    # If no name was provided, generate one
    else:
      name = f'image_link{cls._count}'
      cls.names.append(name)
      return name

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the ImageLink.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
          f"{current_process().name}.{self.name}")

    if self._logger is not None:
      self._logger.log(log_level, msg)

  def set_buffers(self,
                  name: str,
                  lock: synchronize.RLock,
                  meta: managers.DictProxy,
                  ready: synchronize.Event,
                  info: managers.DictProxy,
                  id_value: sharedctypes.Synchronized) -> None:
    """Allows the upstream Block to share synchronization information with the
    downstream Block.

    Args:
      name: The name of the Shared Memory object that will contain the shared
        images.
      lock: The lock that is used to secure reading and writing.
      meta: The shared dictionary containing the image metadata.
      ready: The Event indicating when the shared buffer is ready to be used.
      info: The shared dictionary containing the image shape and dtype.
      id_value: The shared Value containing the current image ID.
    """

    self._img_buffer_name = name
    self._img_lock = lock
    self._metadata_proxy = meta
    self._buffer_ready = ready
    self._img_info_proxy = info
    self._img_id = id_value

  def get_buffers(self) -> tuple[str,
                                 synchronize.RLock,
                                 managers.DictProxy,
                                 synchronize.Event,
                                 managers.DictProxy,
                                 sharedctypes.Synchronized] | None:
    """Allows the downstream Block to retrieve synchronization information from
    the upstream Block.

    Returns:
      The name of the shared memory, the synchronization lock, the metadata
      dictionary, the readiness event, and the image shape and dtype
      dictionary. If any of these is unavailable, returns :obj:`None`.
    """

    if (self._img_buffer_name is not None
        and self._img_lock is not None
        and self._metadata_proxy is not None
        and self._buffer_ready is not None
        and self._img_info_proxy is not None
        and self._img_id is not None):
      return (self._img_buffer_name, self._img_lock, self._metadata_proxy,
              self._buffer_ready, self._img_info_proxy, self._img_id)
    else:
      return None


def img_link(input_block,
             output_block,
             /, *,
             name: str | None = None):
  """Function linking two Blocks, allowing to send images from one to the
  other.

  It instantiates a :class:`~crappy.links.ImageLink` between two children of
  :class:`~crappy.blocks.Block`. The created Link is unidirectional, from the
  input Block to the output Block.

  Args:
    input_block: The Block sending images through the ImageLink.
    output_block: The Block receiving images through the ImageLink.
    name: Name of the ImageLink, to differentiate it from the others. If no
      specific name is given, the ImageLinks are numbered in the order in
      which they are instantiated in the script.

  .. versionadded:: 2.1.0
  """

  ImageLink(input_block=input_block,
            output_block=output_block,
            name=name)
