# coding: utf-8

from time import time
from typing import Optional
import logging
from math import log2

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from Phidget22.Net import Net, PhidgetServerType
  from Phidget22.Devices.VoltageRatioInput import VoltageRatioInput
  from Phidget22.PhidgetException import PhidgetException
except (ImportError, ModuleNotFoundError):
  Net = OptionalModule('Phidget22')
  PhidgetServerType = OptionalModule('Phidget22')
  VoltageRatioInput = OptionalModule('Phidget22')
  PhidgetException = OptionalModule('Phidget22')


class PhidgetWheatstoneBridge(InOut):
  """This class can read voltage ratio values from a Phidget Wheatstone Bridge.

  It relies on the :mod:`Phidget22` module to communicate with the load cell
  conditioner. It can acquire values up to `50Hz` with possible gain values
  from `1` to `128`.

  .. versionadded:: 2.0.4
  """

  def __init__(self,
               channel: int = 0,
               hardware_gain: int = 1,
               data_rate: float = 50,
               gain: float = 1,
               offset: float = 0,
               remote: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      channel: The index of the channel from which to acquire data, as an
        :obj:`int`. Should be either `0` or `1`.
      hardware_gain: The gain used by the conditioner for data acquisition. The
        higher the gain, the better the resolution but the narrower the range.
        Should be a power of `2` between `1` and `128`.
      data_rate: The number of samples to acquire per second, as an :obj:`int`.
        Should be between `0.017` (one sample per minute) and `50`.
      gain: A gain to apply to the acquired value, following the formula :
        :math:`returned = gain * acquired + offset`
      offset: An offset to apply to the acquired value, following the formula :
        :math:`returned = gain * acquired + offset`
      remote: Set to :obj:`True` to drive the bridge via a network VINT Hub,
        or to :obj:`False` to drive it via a USB VINT Hub.
    """

    self._load_cell: Optional[VoltageRatioInput] = None

    super().__init__()

    if channel not in (0, 1):
      raise ValueError("The channel should be 0 or 1 !")
    self._channel = channel

    if hardware_gain not in (2 ** i for i in range(8)):
      raise ValueError("The hardware gain should be either 1, 2, 4, 8, 16, "
                       "32, 64 or 128 !")
    self._hardware_gain = hardware_gain

    self._data_rate = data_rate
    self._gain = gain
    self._offset = offset
    self._remote = remote

    self._last_ratio: Optional[float] = None

  def open(self) -> None:
    """Sets up the connection to the load cell conditioner as well as the
    various callbacks, and waits for the load cell conditioner to attach."""

    # Setting up the load cell conditioner
    self.log(logging.DEBUG, "Enabling server discovery")
    Net.enableServerDiscovery(PhidgetServerType.PHIDGETSERVER_DEVICEREMOTE)
    self._load_cell = VoltageRatioInput()
    self._load_cell.setChannel(self._channel)

    # Setting the remote or local status
    if self._remote is True:
      self._load_cell.setIsLocal(False)
      self._load_cell.setIsRemote(True)
    else:
      self._load_cell.setIsLocal(True)
      self._load_cell.setIsRemote(False)

    # Setting up the callbacks
    self._load_cell.setOnAttachHandler(self._on_attach)
    self._load_cell.setOnVoltageRatioChangeHandler(self._on_ratio_change)

    # Opening the connection to the load cell conditioner
    try:
      self.log(logging.DEBUG, "Trying to attach the load cell conditioner")
      self._load_cell.openWaitForAttachment(10000)
    except PhidgetException:
      raise TimeoutError("Waited too long for the bridge to attach !")

  def get_data(self) -> Optional[list[float]]:
    """Returns the last known voltage ratio value, adjusted with the gain and
    the offset."""

    if self._last_ratio is not None:
      return [time(), self._gain * self._last_ratio + self._offset]

  def close(self) -> None:
    """Closes the connection to the load cell conditioner."""

    if self._load_cell is not None:
      self._load_cell.close()

  def _on_attach(self, _: VoltageRatioInput) -> None:
    """Callback called when the load cell conditioner attaches to the program.

    Sets the data rate and the hardware gain of the conditioner.
    """

    self.log(logging.INFO, "Load cell conditioner successfully attached")

    # Setting the hardware gain
    self._load_cell.setBridgeGain(int(round(log2(self._hardware_gain), 0) + 1))

    # Setting the data rate
    if not ((min_rate := self._load_cell.getMinDataRate()) <= self._data_rate
            <= (max_rate := self._load_cell.getMaxDataRate())):
      raise ValueError(f"The data rate should be between {min_rate} and "
                       f"{max_rate}, got {self._data_rate} !")
    else:
      self._load_cell.setDataRate(self._data_rate)

  def _on_ratio_change(self, _: VoltageRatioInput, ratio: float) -> None:
    """Callback called when the voltage ratio changes."""

    self.log(logging.DEBUG, f"Voltage ratio changed to {ratio}")
    self._last_ratio = ratio

  def _on_error(self,
                _: VoltageRatioInput,
                error_code: int,
                error: str) -> None:
    """Callback called when the load cell conditioner returns an error."""

    raise RuntimeError(f"Got error with error code {error_code}: {error}")
