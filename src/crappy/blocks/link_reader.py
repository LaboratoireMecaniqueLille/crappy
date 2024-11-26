# coding: utf-8

from typing import Optional
import logging

from .meta_block import Block


class LinkReader(Block):
  """This Block reads and displays the data it receives.
  
  It can display data received from any number of :class:`~crappy.links.Link`.
  For each new received data point, a message is displayed in the console with
  the received values.
  
  This Block is the most basic way of displaying data in Crappy. The 
  :class:`~crappy.blocks.Dashboard` Block can be used for a nicer layout, and 
  the :class:`~crappy.blocks.Grapher` Block should be used for plotting data in
  a persistent way.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Reader* to *LinkReader*
  """

  _index = 0

  def __init__(self,
               name: Optional[str] = None,
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      name: If set, will be displayed to identify the LinkReader. Otherwise, 
        the block is automatically named based on the number of its instances 
        already running.
        
        .. versionchanged:: 1.5.5 renamed from *name* to *reader_name*
        .. versionchanged:: 1.5.5 renamed from *reader_name* to *name*
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
        
        .. versionadded:: 1.5.10
      display_freq: if :obj:`True`, displays the looping frequency of the 
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._reader_name = name if name is not None else f'LinkReader ' \
                                                      f'{self._get_index()}'

  def __new__(cls, *args, **kwargs):
    """Increments the instance counter when instantiating a new LinkReader."""

    cls._index += 1
    return super().__new__(cls)

  @classmethod
  def _get_index(cls) -> int:
    """Returns the current number of instantiates Links, as an :obj:`int`."""

    return cls._index

  def loop(self) -> None:
    """Flushes the incoming :class:`~crappy.links.Link` and displays their
    data."""

    for link_data in self.recv_all_data_raw():
      for dic in (dict(i) for i in
                  zip(*([(key, value) for value in values]
                        for key, values in link_data.items()))):
        self.log(logging.INFO, f'{self._reader_name} got: {dic}')
