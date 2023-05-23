# coding: utf-8

from typing import Optional

from .meta_block import Block


class Sink(Block):
  """This Block drops all the data it receives, and does nothing else.
  
  It is only useful for debugging, e.g. with Blocks like the
  :class:`~crappy.blocks.IOBlock` that have a different behavior when they have
  output Links.
  """

  def __init__(self,
               display_freq: bool = False,
               freq: Optional[float] = 10,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.
    
    Args:
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
    """

    super().__init__()
    self.display_freq = display_freq
    self.freq = freq
    self.debug = debug

  def loop(self) -> None:
    """Drops all the received data."""

    self.recv_all_data_raw()
