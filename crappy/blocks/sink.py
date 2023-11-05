# coding: utf-8

from warnings import warn

from .block import Block


class Sink(Block):
  """Test block used to get data and do nothing."""

  def __init__(self,
               verbose: bool = False,
               freq: float = 10) -> None:
    """Sets the args and initializes the parent class."""
    
    if verbose:
      warn("The verbose argument will be replaced by display_freq and debug "
           "in version 2.0.0", FutureWarning)

    super().__init__()
    self.verbose = verbose
    self.freq = freq

  def loop(self) -> None:
    """Simply drops all received data."""

    self.drop()
