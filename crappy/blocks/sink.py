# coding: utf-8

from .block import Block


class Sink(Block):
  """Test block used to get data and do nothing."""

  def __init__(self,
               verbose: bool = False,
               freq: float = 10) -> None:
    """Sets the args and initializes the parent class."""

    super().__init__()
    self.verbose = verbose
    self.freq = freq

  def loop(self) -> None:
    """Simply drops all received data."""

    self.recv_all_data_raw()
