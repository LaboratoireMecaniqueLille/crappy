# coding: utf-8

import time
from .inout import InOut
from .._global import OptionalModule

try:
  import adafruit_mprls
except (ImportError, ModuleNotFoundError):
  adafruit_mprls = OptionalModule('adafruit_mprls')

try:
  import board
except (ImportError, ModuleNotFoundError):
  board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')


class Mprls(InOut):
  """The Mprls inout is meant for reading pressure from Adafruit's Mprls
  pressure sensor.

  It communicates over I2C with the sensor.
  """

  def __init__(self) -> None:
    """Initializes the parent class."""

    InOut.__init__(self)

  def open(self) -> None:
    """Creates the sensor object."""

    self._mpr = adafruit_mprls.MPRLS(board.I2C(), psi_min=0, psi_max=25)

  def get_data(self) -> list:
    """Reads the pressure value.

    Returns:
      The pressure value in hPa.
    """

    return [time.time(), self._mpr.pressure]

  def close(self) -> None:
    """Nothing to do here."""

    pass
