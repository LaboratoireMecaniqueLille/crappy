# coding: utf-8

from typing import Optional

from .meta_block import Block


class Sink(Block):
  """Test block used to get data and do nothing."""

  def __init__(self,
               display_freq: bool = False,
               freq: Optional[float] = 10,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class."""

    super().__init__()
    self.display_freq = display_freq
    self.freq = freq
    self.debug = debug

  def loop(self) -> None:
    """Simply drops all received data."""

    self.recv_all_data_raw()
