# coding: utf-8

from time import time
from typing import Optional, Any, Union, Literal
from collections.abc import Iterable
from itertools import chain
from dataclasses import dataclass, field
from multiprocessing import current_process
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from labjack import ljm
except (ModuleNotFoundError, ImportError):
  ljm = OptionalModule("ljm", "Please install Labjack LJM and the ljm "
                              "Python module")

# Map of thermocouple types vs indexes for the Labjack T7
thcp_map = {'E': 20, 'J': 21, 'K': 22, 'R': 23, 'T': 24, 'S': 25, 'C': 30}


@dataclass
class _Channel:
  """This class is a simple structure holding all the attributes a Labjack
  channel can have.

  Not all the attributes are used by every channel, but they all have a use for
  at least one type of channel.
  """

  name: Union[str, int]

  direction: bool = True
  dtype: Optional[int] = None
  address: int = 1
  gain: float = 1
  offset: float = 0
  make_zero: bool = False
  range: float = 10
  limits: Optional[tuple[float, float]] = None
  resolution: int = 1
  thermocouple: Optional[Literal['E', 'J', 'K', 'R', 'T', 'S', 'C']] = None
  write_at_open: list[tuple[str, float]] = field(default_factory=list)

  def update(self, dic_in: dict[str, Any]) -> None:
    """Updates the channel keys based on the user input."""

    for key, val in dic_in.items():
      if hasattr(self, key):
        setattr(self, key, val)

      # Handling the case when the user enters a wrong key
      else:
        logger = logging.getLogger(
          f"{current_process().name}.LabjackT7.Channel_{self.name}")
        logger.log(logging.WARNING, f"Unknown channel key : {key}, ignoring")


class LabjackT7(InOut):
  """This InOut allows controlling a Labjack T7 device. It can use any channel
  as input/output.

  The Labjack T7 is a very complete DAQ board. It features several ADC, several
  DAC, as well as multiple GPIOs. It can also read thermocouples, and run LUA
  code on an integrated microcontroller. These features can all be controlled
  from Crappy.

  This class is not capable of streaming. For higher frequency, refer to the
  :class:`~crappy.inout.T7Streamer` class.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Labjack_t7* to *LabjackT7*
  """

  def __init__(self,
               channels: Iterable[dict[str, Any]],
               device: Literal['ANY', 'T7', 'T4', 'DIGIT'] = 'ANY',
               connection: Literal['ANY', 'TCP', 'USB',
                                   'ETHERNET', 'WIFI'] = 'ANY',
               identifier: str = 'ANY',
               write_at_open: Optional[Iterable[tuple]] = None,
               no_led: bool = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) of the
        channels to interface with on the Labjack. Each object in this iterable
        should be a :obj:`dict` representing a single channel, and whose keys
        provide information on the channel to use. Refer to the note below for
        more information on the possible keys.
      device: The type of Labjack to open. Possible values include :
        ::

          'ANY', 'T7', 'T4', 'DIGIT'

        Only tested with `'T7'` in Crappy.
      connection: The type of connection used for interfacing with the Labjack.
        Possible values include :
        ::

          'ANY', 'TCP', 'USB', 'ETHERNET', 'WIFI'

      identifier: Any extra information allowing to further identify the
        Labjack to open, like a serial number, an IP address, or a device name.
      write_at_open: If specific names or registers have to be written when
        opening the channel, they can be given here as an iterable (like a
        :obj:`list` or a :obj:`tuple`) of :obj:`tuple`. They will be written in
        the same order as in the given iterable. Refer to the note below for
        the accepted formats.
      no_led: If :obj:`True`, turns off the LED on the Labjack. This led can
        generate noise on the channels `AIN0` and `AIN1`.

    Note:
      - ``channels`` keys:

        - name: The name of the channel to interface with, as written on the
          Labjack's case. Ex: `'AIN0'`. The available settings as well as the
          direction of the channel depend on the given name.

          The name can be:
            - `AINx`: Analog input, linked to an ADC. A gain and offset can be
              provided, and the range and resolution can be adjusted. It is
              always an input. These channels can also be used with
              thermocouples (see below).
            - `(T)DACx`: Analog output, linked to a DAC. A gain and an offset
              can be specified. It is always an output.
            - `(E/F/C/M IOx)`: Digital inputs/outputs. A gain and an offset
              can be specified. It can be either an input or an output, the
              default is output.

        - gain: If the channel is an input, the measured value will be
          modified directly by the Labjack as follows :
          :math:`returned\\_value = gain * measured\\_value + offset`. If the
          channel is an output, the command value will be modified in Crappy as
          follows before being sent to the Labjack :
          :math:`sent\\_value = gain * command + offset`.
        - offset: If the channel is an input, the measured value will be
          modified directly by the Labjack as follows :
          :math:`returned\\_value = gain * measured\\_value + offset`. If the
          channel is an output, the command value will be modified in Crappy as
          follows before being sent to the Labjack :
          :math:`sent\\_value = gain * command + offset`.
        - make_zero: If :obj:`True`, data will be acquired on this channel
          before the test starts, and a compensation value will be deduced
          so that the offset of this channel is `0`. The compensation is
          performed directly by the Labjack. This setting only has effect for
          `AIN` channels defined as inputs. **It will only take effect if the**
          ``make_zero_delay`` **argument of the**
          :class:`~crappy.blocks.IOBlock` **controlling the Labjack is set** !

        - direction: If :obj:`True`, the channel is considered as an output,
          else as an input. Only has effect for `IO` channels, the default is
          output.

        - resolution: The resolution of the acquisition as an integer, refer to
          Labjack documentation for more details. The higher this value the
          better the resolution, but the lower the speed. The possible range is
          either `1` to `8` or to `12` depending on the model. The default is
          `1`. Only has effect for `AIN` channels.

        - range: The range of the acquisition in Volts. A range of `x` means
          that values can be read  between `-x` and `x` Volts. The possible
          values are :
          ::

            0.01, 0.1, 1, 10

          Only has effect for `AIN` channels.

        - limits: A :obj:`tuple` containing the minimum and maximum allowed
          command values to set for this channel. After applying the gain and
          offset to the command, it is then clamped between these two values
          before being sent to the Labjack. Only has effect for output
          channels.

        - thermocouple: The type of thermocouple to read data from. Possible
          values are:
          ::

            'E', 'J', 'K', 'R', 'T', 'S', 'C'

          If specified, it will use the EF to read a temperature directly from
          the thermocouples. Only has effect for `AIN` channels.

        - write_at_open: A :obj:`list` containing commands for writing specific
          names or registers when opening the channel. The commands should be
          given as :obj:`tuple` either in the format
          `(name (str), value (int/float))` or
          `(register (int), type (int), value (float/int))`.

    Warning:
      Do not consider the ``limits`` key as a safety feature. It *should* not
      go beyond/below the given values, but this is not meant to replace
      hardware safety !
    """

    self._handle = None

    super().__init__()

    # Identifiers for the device to open
    self._device = device
    self._connection = connection
    self._identifier = identifier

    # List of commands to send when opening the Labjack
    self._write_at_open = [] if write_at_open is None else list(write_at_open)
    if no_led:
      self._write_at_open.append(('POWER_LED', 0))

    self._channels_in = list()
    self._channels_out = list()

    # Parsing the setting dict given for each channel
    for channel in channels:

      # Checking that the name was given as it's the most important attribute
      if 'name' not in channel:
        raise AttributeError("The given channels must contain the 'name' "
                             "key !")
      name = channel['name']

      # Modbus registers
      if isinstance(name, int):
        chan = _Channel(name=name, dtype=ljm.constants.FLOAT32)
        chan.update(channel)

        # Can be either input or output, the user has to specify
        if chan.direction:
          self._channels_out.append(chan)
        else:
          self._channels_in.append(chan)

      # Analog inputs
      elif isinstance(name, str) and name.startswith('AIN'):
        chan = _Channel(name=name)
        chan.update(channel)

        # Setting the range and the resolution
        chan.write_at_open.extend([
          (*self._parse(f'{name}_RANGE'), chan.range),
          (*self._parse(f'{name}_RESOLUTION_INDEX'), chan.resolution)])

        # Specific commands for thermocouples
        if chan.thermocouple is not None:
          chan.write_at_open.extend([
            (*self._parse(f'{name}_EF_INDEX'), thcp_map[chan.thermocouple]),
            (*self._parse(f'{name}_EF_CONFIG_A'), 1),
            (*self._parse(f'{name}_EF_CONFIG_B'), 60052),
            (*self._parse(f'{name}_EF_CONFIG_D'), 1),
            (*self._parse(f'{name}_EF_CONFIG_E'), 0)])

          chan.address, chan.dtype = self._parse(f'{name}_EF_READ_A')

        # Simplest case with no gain and offset
        elif chan.gain == 1 and chan.offset == 0 and not chan.make_zero:
          chan.address, chan.dtype = self._parse(name)

        # Setting the gain and offset if provided
        else:
          chan.write_at_open.extend([
            (*self._parse(f'{name}_EF_INDEX'), 1),
            (*self._parse(f'{name}_EF_CONFIG_D'), chan.gain),
            (*self._parse(f'{name}_EF_CONFIG_E'), chan.offset
             if not chan.make_zero else 0)])

          chan.address, chan.dtype = self._parse(f'{name}_EF_READ_A')

        self._channels_in.append(chan)

      # Digital to analog converters
      elif isinstance(name, str) and 'DAC' in name:
        chan = _Channel(name=name)
        chan.update(channel)

        chan.address, chan.dtype = self._parse(name)
        self._channels_out.append(chan)

      # Digital inputs and outputs
      elif isinstance(name, str) and 'IO' in name:
        chan = _Channel(name=name)
        chan.update(channel)

        chan.address, chan.dtype = self._parse(name)

        # Can be either input or output, the user has to specify
        if chan.direction:
          self._channels_out.append(chan)
        else:
          self._channels_in.append(chan)

      else:
        raise AttributeError(f"Invalid chan name: {name}")

    self.log(logging.DEBUG, f"Input channels: {self._channels_in}")
    self.log(logging.DEBUG, f"Output channels: {self._channels_out}")

    # Extracting the addresses and data types from all channels
    self._read_addresses = [chan.address for chan in self._channels_in]
    self._read_types = [chan.dtype for chan in self._channels_in]
    self._write_addresses = [chan.address for chan in self._channels_out]
    self._write_types = [chan.dtype for chan in self._channels_out]

    # These attributes will come in use later
    self._last_sent_val = [None for _ in self._write_addresses]

  def open(self) -> None:
    """Opens the Labjack, parses the commands to write at open, and sends
    them."""

    # Opening the Labjack
    self.log(logging.INFO, "Opening the connection to the Labjack")
    self._handle = ljm.openS(self._device, self._connection, self._identifier)

    # Gathering all the data to write at open
    self._write_at_open.extend(chain(*(chan.write_at_open for chan
                                       in chain(self._channels_in,
                                                self._channels_out))))

    # Parsing all the commands given as a tuple of two values
    self._write_at_open = [t if len(t) == 3 else (*self._parse(t[0]), t[1])
                           for t in self._write_at_open]

    # Getting the registers, data types and values in separate tuples
    if self._write_at_open:
      reg, types, values = tuple(zip(*self._write_at_open))
    else:
      reg = types = values = None

    # Finally, writing the commands to write at open
    if reg is not None:
      self.log(logging.DEBUG, f"Writing values {values} to addresses {reg}")
      ljm.eWriteAddresses(handle=self._handle,
                          numFrames=len(reg),
                          aAddresses=reg,
                          aDataTypes=types,
                          aValues=values)

  def make_zero(self, delay: float) -> None:
    """Overriding the method of the parent class, because the Labjack T7
    allows setting offsets directly on the board.

    Setting the offsets on the Labjack is slightly quicker than correcting the
    received values afterward.

    Args:
      delay: The delay during which the data should be acquired for determining
        the offset.
    
    .. versionadded:: 1.5.10
    """

    # No need to acquire data if no channel should be zeroed
    # Also, the IO channels shouldn't be zeroed
    if any(chan.make_zero and 'AIN' in chan.name
           for chan in self._channels_in):

      # Acquiring the data
      super().make_zero(delay)

      # Proceed only if the acquisition went fine
      if self._compensations:
        names, values = tuple(zip(*(
          (f"{chan.name}_EF_CONFIG_E", chan.offset + off)
          for off, chan in zip(self._compensations, self._channels_in)
          if chan.make_zero and 'AIN' in chan.name)))

        # Setting the offsets on the Labjack
        self.log(logging.DEBUG, f"Writing values {values} to registers "
                                f"{names}")
        ljm.eWriteNames(handle=self._handle,
                        numFrames=len(names),
                        aNames=names,
                        aValues=values)

        # Resetting the software offsets to avoid double compensation
        self._compensations = list()

  def get_data(self) -> list[float]:
    """Reads the signal on all pre-defined input channels, and returns the
    values along with a timestamp."""

    return [time()] + ljm.eReadAddresses(handle=self._handle,
                                         numFrames=len(self._read_addresses),
                                         aAddresses=self._read_addresses,
                                         aDataTypes=self._read_types)

  def set_cmd(self, *cmd: float) -> None:
    """Sets the tension commands on the output channels.

    The given gain and offset are first applied, then the commands are clamped
    to the given limits. The commands are then written to the Labjack, only if
    they differ from the last written ones.
    """

    # First, applying the given gain and offsets to the commands
    cmd = [chan.gain * val + chan.offset
           for val, chan in zip(cmd, self._channels_out)]

    # Then, clamping the commands if limits were given
    cmd = [val if chan.limits is None
           else max(min(val, chan.limits[1]), chan.limits[0])
           for val, chan in zip(cmd, self._channels_out)]

    # Checking which values have to be updated and which are unchanged
    to_upd = [not val == prev for val, prev in zip(cmd, self._last_sent_val)]

    # Updating the list of last sent values
    self._last_sent_val = list(cmd)

    # No use continuing if there's nothing to update
    if any(to_upd):

      # Getting the addresses, data types and values to send
      addresses, types, values = tuple(zip(*(
        (addr, typ, val) for addr, typ, val, upd in
        zip(self._write_addresses, self._write_types, cmd, to_upd) if upd)))

      # Sending the commands
      if addresses:
        self.log(logging.DEBUG, f"Writing values {values} to addresses "
                                f"{addresses}")
        ljm.eWriteAddresses(handle=self._handle,
                            numFrames=len(addresses),
                            aAddresses=addresses,
                            aDataTypes=types,
                            aValues=values)

  def close(self) -> None:
    """Closes the connection to the Labjack."""

    if self._handle is not None:
      self.log(logging.INFO, "Closing the connection to the Labjack")
      ljm.close(self._handle)

  @staticmethod
  def _parse(name: str) -> tuple[int, int]:
    """Wrapper around :meth:`ljm.nameToAddress` to make the code clearer."""

    return ljm.nameToAddress(name)
