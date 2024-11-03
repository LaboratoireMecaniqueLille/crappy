# coding: utf-8

from enum import IntEnum
from collections import namedtuple
from struct import calcsize, unpack, pack
from typing import Union, Optional, Literal
from collections.abc import Callable
from multiprocessing import current_process
import logging

from .i2c_message import I2CMessage
from ..._global import OptionalModule
try:
  from usb import util
  from usb.core import find, USBError

  Ftdi_req_out = util.build_request_type(util.CTRL_OUT,
                                         util.CTRL_TYPE_VENDOR,
                                         util.CTRL_RECIPIENT_DEVICE)

  Ftdi_req_in = util.build_request_type(util.CTRL_IN,
                                        util.CTRL_TYPE_VENDOR,
                                        util.CTRL_RECIPIENT_DEVICE)

except (ModuleNotFoundError, ImportError):
  util = OptionalModule("pyusb")
  find = OptionalModule("pyusb")
  USBError = OptionalModule("pyusb")
  Ftdi_req_out = 0x40
  Ftdi_req_in = 0xC0

ft232h_cmds = {'write_bytes_PVE_MSB': 0x10,
               'write_bytes_NVE_MSB': 0x11,
               'write_bits_NVE_MSB': 0x13,
               'write_bytes_PVE_LSB': 0x18,
               'write_bytes_NVE_LSB': 0x19,
               'read_bytes_PVE_MSB': 0x20,
               'read_bits_PVE_MSB': 0x22,
               'read_bytes_NVE_MSB': 0x24,
               'read_bytes_PVE_LSB': 0x28,
               'read_bytes_NVE_LSB': 0x2C,
               'rw_bytes_PVE_NVE_MSB': 0x31,
               'rw_bytes_NVE_PVE_MSB': 0x34,
               'rw_bytes_PVE_NVE_LSB': 0x39,
               'rw_bytes_NVE_PVE_LSB': 0x3C,
               'set_bits_low': 0x80,
               'get_bits_low': 0x81,
               'set_bits_high': 0x82,
               'get_bits_high': 0x83,
               'loopback_start': 0x84,
               'loopback_end': 0x85,
               'set_tck_divisor': 0x86,
               'send_immediate': 0x87,
               'disable_clk_div5': 0x8A,
               'enable_clk_div5': 0x8B,
               'enable_clk_3phase': 0x8C,
               'disable_clk_3phase': 0x8D,
               'enable_clk_adaptative': 0x96,
               'disable_clk_adaptative': 0x97,
               'drive_zero': 0x9E}

ft232h_sio_req = {'reset': 0x00,
                  'set_event_char': 0x06,
                  'set_error_char': 0x07,
                  'set_latency_timer': 0x09,
                  'set_bitmode': 0x0B,
                  'read_eeprom': 0x90,
                  'write_eeprom': 0x91}

ft232h_sio_args = {'reset': 0,
                   'purge_RX': 1,
                   'purge_TX': 2}

Ftdi_vendor_id = 0x0403
ft232h_product_id = 0x6014

ft232h_latency = {'min': 1,
                  'max': 255}

ft232h_clock = {'base': 6.0E6,
                'high': 30.0E6}

ft232h_tx_empty_bits = 0x60
ft232h_max_payload = 0xFF
ft232h_mpsse_bit_delay = 0.5E-6
ft232h_port_width = 16
ft232h_eeprom_size = 256

ft232h_eeprom = {'has_serial_pos': 0x0A,
                 'str_table': 0x0E,
                 'str_position': 0xA0}

ft232h_pins = {'SCL': 0x01,
               'SDAO': 0x02,
               'SDAI': 0x04,
               'SCL_FB': 0x80,
               'SCK': 0x01,
               'DO': 0x02,
               'DI': 0x04,
               'CS': 0x08}

ft232h_i2c_timings = namedtuple('I2CTimings',
                                't_hd_sta t_su_sta t_su_sto t_buf')

ft232h_modes = ['SPI', 'I2C', 'GPIO_only', 'Write_serial_nr']

ft232h_pin_nr = {pin: index for pin, index in zip(
  ['D{}'.format(i) for i in range(8)] +
  ['C{}'.format(i) for i in range(8)], [2 ** j for j in range(16)])}

ft232h_i2c_speed = {100E3: ft232h_i2c_timings(4.0E-6, 4.7E-6, 4.0E-6, 4.7E-6),
                    400E3: ft232h_i2c_timings(0.6E-6, 0.6E-6, 0.6E-6, 1.3E-6),
                    1E6: ft232h_i2c_timings(0.26E-6, 0.26E-6, 0.26E-6, 0.5E-6)}


class FindSerialNumber:
  """A class used for finding USB devices matching a given serial number, using
     the usb.core.find method.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Find_serial_number* to *FindSerialNumber*
  """

  def __init__(self, serial_number: str) -> None:
    self.serial_number = serial_number

  def __call__(self, device) -> bool:
    return device.serial_number == self.serial_number


class FT232H:
  """A class for controlling FTDI's USB to Serial FT232H.

  Communication in SPI and I2C are implemented, along with GPIO control. The
  name of the methods for SPI and I2C communication are those of :mod:`smbus`
  and :mod:`spidev` libraries, in order to facilitate the use and the
  integration in a multi-backend environment. This class also allows to write a
  USB serial number in the EEPROM, as there's no default serial number on the
  chip.

  Note:
    The FT232H does not support clock stretching and this may cause bugs with
    some I2C devices. Lowering the ``i2c_speed`` may solve the problem.

  Important:
    If using Adafruit's board, its `I2C Mode` switch should of course be set to
    the correct value according to the chosen mode.

  Important:
    **Only for Linux users:** In order to drive the FT232H, the appropriate
    udev rule should be set. This can be done using the `udev_rule_setter`
    utility in ``crappy``'s `util` folder. It is also possible to add it
    manually by running:
    ::

      echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"0403\\", \
MODE=\\"0666\\\"" | sudo tee ftdi.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.

  Important:
    For controlling several FT232H from the same computer, it is first
    necessary to set their USB serial numbers. Otherwise, an error will be
    raised. This can be done using the crappy utility
    ``set_ft232h_serial_nr.py``.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *ft232h* to *FT232H*
  """

  class BitMode(IntEnum):
    """Commands for changing the bit mode."""

    RESET = 0x00  # switch off alternative mode (default to UART)
    BITBANG = 0x01  # classical asynchronous bitbang mode
    MPSSE = 0x02  # MPSSE mode, available on 2232x chips
    SYNCBB = 0x04  # synchronous bitbang mode
    MCU = 0x08  # MCU Host Bus Emulation mode,
    OPTO = 0x10  # Fast Opto-Isolated Serial Interface Mode
    CBUS = 0x20  # Bitbang on CBUS pins of R-type chips
    SYNCFF = 0x40  # Single Channel Synchronous FIFO mode

  def __init__(self,
               mode: Literal['SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'],
               serial_nr: Optional[str] = None,
               i2c_speed: float = 100E3,
               spi_turbo: bool = False) -> None:
    """Checks the arguments validity, initializes the device and sets the
    locks.

    Args:
      mode: The communication mode as a :obj:`str`, can be :
        ::

          'SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'

        GPIOs can be driven in any mode, but faster speeds are achievable in
        `GPIO_only` mode.
      serial_nr: The serial number of the FT232H to drive, as a :obj:`str`. In
        `Write_serial_nr` mode, the serial number to be written.
      i2c_speed: In I2C mode, the I2C bus clock frequency in Hz, as an
        :obj:`int`. Available values are :
        ::

          100E3, 400E3, 1E6

        or any value between `10kHz` and `100kHz`. Lowering below the default
        value may solve I2C clock stretching issues on some devices.
      spi_turbo: If :obj:`True`, increases the achievable bus speed in SPI
        mode, but may not work with some devices.

    Note:
      - **CS pin**:
        The CS pin for selecting SPI devices is always `D3`. This pin is
        reserved and cannot be used as a GPIO. If you want to drive the CS line
        manually, it is possible not to drive the CS pin by setting the SPI
        parameter :attr:`no_cs` to :obj:`True` and to drive the CS line from a
        GPIO instead.

      - ``mode``:
        It is not possible to simultaneously control slaves over SPI and I2C,
        due to different hardware requirements for the two protocols. Trying to
        do so will most likely raise an error or lead to inconsistent behavior.

    """

    if mode not in ft232h_modes:
      raise ValueError("mode should be in {}".format(ft232h_modes))
    self._ft232h_mode = mode

    if mode == 'Write_serial_nr' and serial_nr is None:
      raise ValueError("Cannot set serial number if it is not specified !")

    if i2c_speed not in ft232h_i2c_speed:
      try:
        if not 10E3 <= i2c_speed < 100E3:
          raise ValueError("i2c_speed should be in {} or between 10E3 and "
                           "100E3".format(list(ft232h_i2c_speed.values())))
      except TypeError:
        raise TypeError("i2c_speed should be a float or an int !")

    self._gpio_low = 0
    self._gpio_high = 0
    self._gpio_dir = 0
    self._retry_count = 16

    self._usb_write_timeout = 5000
    self._usb_read_timeout = 5000

    self._serial_nr = serial_nr
    self._turbo = spi_turbo
    self._i2c_speed = i2c_speed

    self._nb_attempt_1 = 8
    self._nb_attempt_2 = 8

    self._bits_per_word = 8
    self._cshigh = False
    self._no_cs = False
    self._loop = False
    self._lsbfirst = False
    self._max_speed_hz = 400E3
    self._mode = 0
    self._threewire = False
    self._spi_param_changed = True

    self._logger: Optional[logging.Logger] = None

    self._initialize()

    if mode == 'Write_serial_nr':
      self.log(logging.WARNING, f"Setting the FT232H seria lnumber to "
                                f"{serial_nr}")
      self._set_serial_number(serial_nr)
      self.close()

  def _initialize(self) -> None:
    """Initializing the FT232H according to the chosen mode.

    The main differences are for the choice of the clock frequency and
    parameters.
    """

    # FT232H properties
    fifo_sizes = (1024, 1024)
    latency = 16

    # I2C properties
    if self._ft232h_mode == 'I2C':
      timings = ft232h_i2c_speed[self._i2c_speed if self._i2c_speed in
                                 ft232h_i2c_speed else 100E3]
      frequency = self._i2c_speed

      self._ck_hd_sta = self._compute_delay_cycles(timings.t_hd_sta)
      self._ck_su_sto = self._compute_delay_cycles(timings.t_su_sto)
      ck_su_sta = self._compute_delay_cycles(timings.t_su_sta)
      ck_buf = self._compute_delay_cycles(timings.t_buf)
      self._ck_idle = max(ck_su_sta, ck_buf)
      self._ck_delay = ck_buf

      self._i2c_mask = ft232h_pins['SCL'] | ft232h_pins['SDAO'] | \
          ft232h_pins['SDAI']
      self._i2c_dir = ft232h_pins['SCL'] | ft232h_pins['SDAO']

    # SPI properties
    elif self._ft232h_mode == 'SPI':
      frequency = 400E3

      self._cs_bit = ft232h_pins['CS']
      self._spi_dir = self._cs_bit | ft232h_pins['SCK'] | ft232h_pins['DO']
      self._spi_mask = self._cs_bit | ft232h_pins['SCK'] | \
          ft232h_pins['DO'] | ft232h_pins['DI']

    else:
      frequency = 400E3

    # Finding the matching USB devices
    if self._serial_nr is not None and self._ft232h_mode != 'Write_serial_nr':
      devices = find(find_all=True,
                     idVendor=Ftdi_vendor_id,
                     idProduct=ft232h_product_id,
                     custom_match=FindSerialNumber(self._serial_nr))
    else:
      devices = find(find_all=True,
                     idVendor=Ftdi_vendor_id,
                     idProduct=ft232h_product_id)

    # Checking if there's only 1 device matching
    if len(devices := list(devices)) == 0:
      raise IOError("No matching ft232h connected")
    elif len(devices) > 1:
      raise IOError("Several ft232h devices found, please specify a serial_nr")
    else:
      self._usb_dev = devices[0]
      self.log(logging.DEBUG, f"USB device found: {self._usb_dev}")

    try:
      self._serial_nr = self._usb_dev.serial_number
    except ValueError:
      self._serial_nr = ""

    # Configuring the USB device, interface and endpoints
    try:
      if self._usb_dev.is_kernel_driver_active(0):
        self._usb_dev.detach_kernel_driver(0)
        self.log(logging.INFO, "Setting USB configuration for the FT232H")
      self._usb_dev.set_configuration()
    except USBError:
      self.log(logging.ERROR,
               "Could not set USB device configuration !\nYou may have to "
               "install the udev-rules for this USB device, this can be done "
               "using the udev_rule_setter utility in the util folder")
      raise
    config = self._usb_dev.get_active_configuration()

    interface = config[(0, 0)]
    self._index = interface.bInterfaceNumber + 1
    endpoints = sorted([ep.bEndpointAddress for ep in interface])
    self._in_ep, self._out_ep = endpoints[:2]

    endpoint = interface[0]
    self._max_packet_size = endpoint.wMaxPacketSize

    # Invalidate data in the readbuffer
    self._readoffset = 0
    self._readbuffer = bytearray()
    # Drain input buffer
    self._purge_buffers()
    # Shallow reset

    if self._ctrl_transfer_out(ft232h_sio_req['reset'],
                               ft232h_sio_args['reset']):
      raise IOError('Unable to reset FTDI device')
    # Reset feature mode
    self._set_bitmode(0, FT232H.BitMode.RESET)

    # Set latency timer
    self._set_latency_timer(latency)

    # Set chunk size and invalidate all remaining data
    self._writebuffer_chunksize = fifo_sizes[0]
    self._readoffset = 0
    self._readbuffer = bytearray()
    self._readbuffer_chunksize = min(fifo_sizes[0], fifo_sizes[1],
                                     self._max_packet_size)

    # Reset feature mode
    self._set_bitmode(0, FT232H.BitMode.RESET)
    # Drain buffers
    self._purge_buffers()
    # Disable event and error characters
    if self._ctrl_transfer_out(ft232h_sio_req['set_event_char'], 0):
      raise IOError('Unable to set event char')
    if self._ctrl_transfer_out(ft232h_sio_req['set_error_char'], 0):
      raise IOError('Unable to set error char')

    # Enable MPSSE mode
    if self._ft232h_mode == 'GPIO_only':
      self.log(logging.DEBUG, "Setting the mode to GPIO_only")
      self._set_bitmode(0xFF, FT232H.BitMode.MPSSE)
    else:
      self.log(logging.DEBUG, f"Setting the mode to {self._ft232h_mode}")
      self._set_bitmode(self._direction, FT232H.BitMode.MPSSE)

    # Configure clock
    if self._ft232h_mode == 'I2C':
      # Note that bus frequency may differ from clock frequency, when
      # 3-phase clock is enabled
      self._set_frequency(3 * frequency / 2)
    else:
      self._set_frequency(frequency)

    # Configure pins
    self.log(logging.DEBUG, "Configuring the FT232H pins")
    if self._ft232h_mode == 'I2C':
      cmd = bytearray(self._idle)
      cmd.extend((ft232h_cmds['set_bits_high'], 0, 0))
      self._write_data(cmd)
    elif self._ft232h_mode == 'SPI':
      cmd = bytearray((ft232h_cmds['set_bits_low'],
                       self._cs_bit & 0xFF,
                       self._direction & 0xFF))
      cmd.extend((ft232h_cmds['set_bits_high'],
                  (self._cs_bit >> 8) & 0xFF,
                  (self._direction >> 8) & 0xFF))
      self._write_data(cmd)
    else:
      cmd = bytearray((ft232h_cmds['set_bits_low'], 0, 0))
      cmd.extend((ft232h_cmds['set_bits_high'], 0, 0))
      self._write_data(cmd)

    # Disable loopback
    self.log(logging.DEBUG, "Disabling loopback")
    self._write_data(bytearray((ft232h_cmds['loopback_end'],)))
    # Validate MPSSE
    bytes_ = bytes(self._read_data_bytes(2))
    if (len(bytes_) >= 2) and (bytes_[0] == '\xfa'):
      raise IOError("Invalid command @ %d" % bytes_[1])

    # I2C-specific settings
    if self._ft232h_mode == 'I2C':
      self.log(logging.DEBUG, "Configuring I2C-specific features")
      self._tx_size, self._rx_size = fifo_sizes

      # Enable 3-phase clock
      self._write_data(bytearray([True and
                                  ft232h_cmds['enable_clk_3phase'] or
                                  ft232h_cmds['disable_clk_3phase']]))

      # Enable drivezero mode
      self._write_data(bytearray([ft232h_cmds['drive_zero'],
                                  self._i2c_mask & 0xFF,
                                  (self._i2c_mask >> 8) & 0xFF]))

    # Disable adaptative clock
    self._write_data(bytearray([False and
                                ft232h_cmds['enable_clk_adaptative'] or
                                ft232h_cmds['disable_clk_adaptative']]))

  def log(self, level: int, msg: str) -> None:
    """Wrapper for logging messages.

    Also initializes the Logger on the first message.

    Args:
      level: The logging level of the message, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(f"{current_process().name}."
                                       f"{type(self).__name__}")

    self._logger.log(level, msg)

  @staticmethod
  def _compute_delay_cycles(value: float) -> int:
    """Approximates the number of clock cycles over a given delay.

    Args:
      value: The delay in seconds, as a :obj:`float`.

    Returns:
      The number of clock cycles, as an :obj:`int`.
    """

    bit_delay = ft232h_mpsse_bit_delay
    return max(1, int((value + bit_delay) / bit_delay))

  def _set_latency_timer(self, latency: int) -> None:
    """Sets the latency timer.

    Sets the latency timer, i.e. the delay the chip waits before sending the
    data stored in the buffer to the host. Not applicable when a send
    immediate command is issued, in which case the buffered data is returned
    immediately.

    Args:
      latency: The latency in milliseconds, as an :obj:`int`.
    """

    self.log(logging.DEBUG, f"Setting the latency timer to {latency}")

    if not ft232h_latency['min'] <= latency <= ft232h_latency['max']:
      raise ValueError("Latency out of range")
    if self._ctrl_transfer_out(ft232h_sio_req['set_latency_timer'], latency):
      raise IOError('Unable to set latency timer')

  def _set_frequency(self, frequency: float) -> float:
    """Sets the bus frequency.

    Sets the FT232H clock divisor value, according to the desired bus
    frequency. The actual bus frequency is then as close as possible to the
    desired value, but may still be slightly different.

    Args:
      frequency: The desired bus frequency in Hz, as a :obj:`float`.

    Returns:
      The actual bus frequency, as a :obj:`float`.
    """

    self.log(logging.DEBUG, f"Setting the clock frequency to {frequency}")

    # Calculate base speed clock divider
    divcode = ft232h_cmds['enable_clk_div5']
    divisor = int((ft232h_clock['base'] + frequency / 2) / frequency) - 1
    divisor = max(0, min(0xFFFF, divisor))
    actual_freq = ft232h_clock['base'] / (divisor + 1)
    error = (actual_freq / frequency) - 1

    # Calculate high speed clock divider
    divisor_hs = int((ft232h_clock['high'] + frequency / 2) / frequency) - 1
    divisor_hs = max(0, min(0xFFFF, divisor_hs))
    actual_freq_hs = ft232h_clock['high'] / (divisor_hs + 1)
    error_hs = (actual_freq_hs / frequency) - 1

    # Enable if closer to desired frequency
    # =====================================================================
    if abs(error_hs) <= abs(error):
      divcode = ft232h_cmds['disable_clk_div5']
      divisor = divisor_hs
      actual_freq = actual_freq_hs

    # FTDI expects little endian
    cmd = bytearray((divcode,))
    cmd.extend((ft232h_cmds['set_tck_divisor'], divisor & 0xff,
                (divisor >> 8) & 0xff))
    cmd.extend((ft232h_cmds['send_immediate'],))
    self._write_data(cmd)

    # validate MPSSE
    bytes_ = bytes(self._read_data_bytes(2))
    if (len(bytes_) >= 2) and (bytes_[0] == '\xfa'):
      raise IOError("Invalid command @ %d" % bytes_[1])

    # Drain input buffer
    self._purge_rx_buffer()

    return actual_freq

  def _set_bitmode(self, bitmask: int, mode: BitMode) -> None:
    """Sets the bitbang mode.

    Args:
      bitmask: Mask for choosing the driven GPIOs.
      mode: The bitbang mode to be set.
    """

    mask = sum(FT232H.BitMode)
    value = (bitmask & 0xff) | ((mode.value & mask) << 8)
    if self._ctrl_transfer_out(ft232h_sio_req['set_bitmode'], value):
      raise IOError('Unable to set bitmode')

  def _purge_buffers(self) -> None:
    """Clears the buffers on the chip and the internal read buffer."""

    self._purge_rx_buffer()
    self._purge_tx_buffer()

  def _purge_rx_buffer(self) -> None:
    """Clears the USB receive buffer on the chip (host-to-ftdi) and the
       internal read buffer."""

    if self._ctrl_transfer_out(ft232h_sio_req['reset'],
                               ft232h_sio_args['purge_RX']):
      raise IOError('Unable to flush RX buffer')
    # Invalidate data in the readbuffer
    self._readoffset = 0
    self._readbuffer = bytearray()

  def _purge_tx_buffer(self) -> None:
    """Clears the USB transmit buffer on the chip (ftdi-to-host)."""

    if self._ctrl_transfer_out(ft232h_sio_req['reset'],
                               ft232h_sio_args['purge_TX']):
      raise IOError('Unable to flush TX buffer')

  def _ctrl_transfer_out(self,
                         reqtype: int,
                         value: int,
                         data: bytes = b'') -> int:
    """Sends a control message to the device.

    Args:
      reqtype: bmRequest
      value: wValue
      data: payload

    Returns:
      Number of bytes actually written
    """

    try:
      self.log(logging.DEBUG,
               f"Sending USB control transfer with request type {Ftdi_req_out}"
               f", request {reqtype}, value {value}, index {self._index}, "
               f"data {data}")
      return self._usb_dev.ctrl_transfer(
        Ftdi_req_out, reqtype, value, self._index,
        bytearray(data), self._usb_write_timeout)
    except USBError as ex:
      raise IOError('UsbError: %s' % str(ex))

  def _set_serial_number(self, serial_number: str) -> None:
    """(Over)Writes the serial number.

    Writes the desired serial number to the EEPROM. It is then accessible to
    USB commands as a string descriptor. Also sets the manufacturer and
    product string descriptors.

    Args:
      serial_number: Serial number to be written in the EEPROM, as a
        :obj:`str`.
    """

    if not isinstance(serial_number, str):
      serial_number = str(serial_number)
    if any(char in serial_number for char in ':/'):
      raise ValueError("Invalid character : or / in serial number")

    # Reading current eeprom
    word_count = round(ft232h_eeprom_size / 2)
    word_addr = 0
    data = bytearray()
    while word_count:
      try:
        self.log(logging.DEBUG,
                 f"Sending USB control transfer with request type "
                 f"{Ftdi_req_in}, request {ft232h_sio_req['read_eeprom']}, "
                 f"value 0, index {word_addr}, data 2")
        buf = self._usb_dev.ctrl_transfer(
          Ftdi_req_in, ft232h_sio_req['read_eeprom'], 0,
          word_addr, 2, self._usb_read_timeout)
        self.log(logging.DEBUG, f"Read {buf} from the USB device")
      except USBError as exc:
        raise IOError('UsbError: %s' % exc) from exc
      if not buf:
        raise IOError('EEPROM read error @ %d' % (word_addr << 1))
      data.extend(buf)
      word_count -= 1
      word_addr += 1
    new_eeprom = data[0:ft232h_eeprom_size]

    # Setting the has_serial flag to True
    new_eeprom[ft232h_eeprom['has_serial_pos']] |= 1 << 3

    # Changing the string descriptors and the descriptors index
    str_descriptors = {'manufacturer': 'FTDI',
                       'product': 'FT232H',
                       'serial': serial_number}
    stream = bytearray()
    str_pos = ft232h_eeprom['str_position']
    tbl_pos = ft232h_eeprom['str_table']
    data_pos = str_pos
    for name in str_descriptors:
      new_str = str_descriptors[name].encode('utf-16le')
      length = len(new_str) + 2
      stream.append(length)
      stream.append(util.DESC_TYPE_STRING)  # string descriptor
      stream.extend(new_str)
      new_eeprom[tbl_pos] = data_pos
      tbl_pos += 1
      new_eeprom[tbl_pos] = length
      tbl_pos += 1
      data_pos += length
    new_eeprom[str_pos:str_pos + len(stream)] = stream

    # Filling the remaining space with zeros
    crc_pos = len(new_eeprom)
    rem = crc_pos - (str_pos + len(stream))
    new_eeprom[str_pos + len(stream):crc_pos] = bytes(rem)

    # Checking the eeprom length
    if len(new_eeprom) != ft232h_eeprom_size:
      raise ValueError("Eeprom_size not matching, serial number may be "
                       "too long, eeprom not written")

    # Calculating the new checksum and modifying the corresponding bytes
    checksum = 0xAAAA
    for idx in range(0, len(new_eeprom[:-2]), 2):
      v = ((new_eeprom[:-2][idx + 1] << 8) + new_eeprom[:-2][idx]) & 0xFFFF
      checksum = v ^ checksum
      checksum = ((checksum << 1) & 0xFFFF) | ((checksum >> 15) & 0xFFFF)

    new_eeprom[-2] = checksum & 0xFF
    new_eeprom[-1] = checksum >> 8

    # Updating the eeprom
    addr = 0
    for word in unpack('<%dH' % (len(new_eeprom) // 2), new_eeprom):
      self.log(logging.DEBUG,
               f"Sending USB control transfer with request type {Ftdi_req_out}"
               f", request {ft232h_sio_req['write_eeprom']}, value {word}, "
               f"index {addr >> 1}, data b''")
      out = self._usb_dev.ctrl_transfer(
        Ftdi_req_out, ft232h_sio_req['write_eeprom'],
        word, addr >> 1, b'', self._usb_write_timeout)
      if out:
        raise IOError('EEPROM Write Error @ %d' % addr)
      addr += 2

  def _write_data(self, data: Union[bytearray, bytes]) -> int:
    """Writes data to the FT232H.

    Writes the sequence of MPSSE commands and data to the FTDI port. Data
    buffer is split into chunk-sized blocks before being sent over the USB bus.

    Args:
      data: The byte stream to send to the FTDI interface

    Returns:
      Count of written bytes
    """

    offset = 0
    size = len(data)
    try:
      while offset < size:
        write_size = self._writebuffer_chunksize
        if offset + write_size > size:
          write_size = size - offset

        try:
          self.log(logging.DEBUG,
                   f"Sending USB write command to endpoint {self._in_ep}"
                   f"and with data {data[offset:offset + write_size]}")
          length = self._usb_dev.write(self._in_ep,
                                       data[offset:offset + write_size],
                                       self._usb_write_timeout)
        except USBError:
          raise

        if length <= 0:
          raise USBError("Usb bulk write error")
        offset += length
      return offset
    except USBError:
      self.log(logging.ERROR, "An error occurred while writing to USB device")
      raise

  def _read_data_bytes(self,
                       size: int,
                       attempt: int = 2,
                       request_gen: Optional[
                         Callable[[int], Union[bytearray,
                                               bytes]]] = None) -> bytes:
    """Reads data from the FT232H.

    Reads data from the FTDI interface. The data buffer is rebuilt from
    chunk-sized blocks received over the USB bus. The FTDI device always sends
    internal status bytes, which are stripped out as not part of the data
    payload.

    Args:
      size: The number of bytes to receive from the device, as an :obj:`int`.
      attempt: Attempt cycle count
      request_gen: A callable that takes the number of bytes read and expects a
        bytes buffer to send back to the remote device. This is only useful to
        perform optimized/continuous transfer from a slave device.

    Returns:
      Payload bytes
    """

    # Packet size sanity check
    if not self._max_packet_size:
      raise ValueError("max_packet_size is bogus")
    packet_size = self._max_packet_size
    length = 1  # initial condition to enter the usb_read loop
    data = bytearray()
    # everything we want is still in the cache?
    if size <= len(self._readbuffer) - self._readoffset:
      data = self._readbuffer[self._readoffset:self._readoffset + size]
      self._readoffset += size
      return data
    # something still in the cache, but not enough to satisfy 'size'?
    if len(self._readbuffer) - self._readoffset != 0:
      data = self._readbuffer[self._readoffset:]
      # end of readbuffer reached
      self._readoffset = len(self._readbuffer)
    # read from USB, filling in the local cache as it is empty
    retry = attempt
    req_size = size
    try:
      while (len(data) < size) and (length > 0):
        while True:

          try:
            self.log(logging.DEBUG,
                     f"Sending USB read command to endpoint {self._out_ep}"
                     f"to read {self._readbuffer_chunksize} bytes")
            tempbuf = self._usb_dev.read(self._out_ep,
                                         self._readbuffer_chunksize,
                                         self._usb_read_timeout)
            self.log(logging.DEBUG, f"Read {tempbuf} from the USB device")
          except USBError:
            raise

          retry -= 1
          length = len(tempbuf)
          # the received buffer contains at least one useful databyte
          # (first 2 bytes in each packet represent the current modem
          # status)
          if length >= 2:
            if tempbuf[1] & ft232h_tx_empty_bits:
              if request_gen:
                if (req_size := req_size - (length - 2)) > 0:
                  if cmd := request_gen(req_size):
                    self._write_data(cmd)
          if length > 2:
            retry = attempt
            # skip the status bytes
            chunks = (length + packet_size - 1) // packet_size
            count = packet_size - 2
            self._readbuffer = bytearray()
            self._readoffset = 0
            srcoff = 2
            for _ in range(chunks):
              self._readbuffer += tempbuf[srcoff:srcoff + count]
              srcoff += packet_size
            length = len(self._readbuffer)
            break
          # received buffer only contains the modem status bytes
          # no data received, may be late, try again
          if retry > 0:
            continue
          # no actual data
          self._readbuffer = bytearray()
          self._readoffset = 0
          # no more data to read?
          return data
        if length > 0:
          # data still fits in buf?
          if (len(data) + length) <= size:
            data += self._readbuffer[self._readoffset:
                                     self._readoffset + length]
            self._readoffset += length
            # did we read exactly the right amount of bytes?
            if len(data) == size:
              return data
          else:
            # partial copy, not enough bytes in the local cache to
            # fulfill the request
            part_size = min(size - len(data),
                            len(self._readbuffer) - self._readoffset)
            if part_size < 0:
              raise ValueError("Internal Error")
            data += self._readbuffer[self._readoffset:
                                     self._readoffset + part_size]
            self._readoffset += part_size
            return data
    except USBError:
      self.log(logging.ERROR, "An error occurred while writing to USB device")
      raise
    # never reached
    raise ValueError("Internal error")

  @property
  def _clk_hi_data_lo(self) -> tuple[int, int, int]:
    """Returns the MPSSE command for driving CLK line high and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SCL'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_input(self) -> tuple[int, int, int]:
    """Returns the MPSSE command for driving CLK line low and listening to SDA
       line, while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            ft232h_pins['SCL'] | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_hi(self) -> tuple[int, int, int]:
    """Returns the MPSSE command for driving CLK line low and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SDAO'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_lo(self) -> tuple[int, int, int]:
    """Returns the MPSSE command for driving CLK line low and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _idle(self) -> tuple[int, int, int]:
    """Returns the MPSSE command for driving CLK line high and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._i2c_dir | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _start(self) -> tuple[int, ...]:
    """Returns the MPSSE command for issuing and I2C start condition."""

    return self._clk_hi_data_lo * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta

  @property
  def _stop(self) -> tuple[int, ...]:
    """Returns the MPSSE command for issuing and I2C stop condition."""

    return self._clk_lo_data_hi * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta + \
        self._clk_hi_data_lo * self._ck_su_sto + \
        self._idle * self._ck_idle

  def _do_prolog(self, i2caddress: int) -> None:
    """Sends the MPSSE commands for starting an I2C transaction.

    Args:
      i2caddress: I2C address of the slave
    """

    if i2caddress is None:
      return
    cmd = bytearray(self._idle * self._ck_delay)
    cmd.extend(self._start)
    cmd.extend((ft232h_cmds['write_bytes_NVE_MSB'], 0, 0))
    cmd.append(i2caddress)
    try:
      self._send_check_ack(cmd)
    except IOError:
      raise

  def _do_write(self, out: list) -> None:
    """Sends the MPSSE commands for writing bytes to an I2C slave.

    Args:
      out: List of bytes to write
    """

    if not isinstance(out, bytearray):
      out = bytearray(out)
    if not out:
      return
    for byte in out:
      cmd = bytearray((ft232h_cmds['write_bytes_NVE_MSB'], 0, 0))
      cmd.append(byte)
      self._send_check_ack(cmd)

  def _do_read(self, readlen: int) -> bytearray:
    """Sends the MPSSE commands for reading bytes from an I2C slave, and then
    returns these bytes.

    Args:
      readlen: Number of bytes to read

    Returns: Read bytes as a bytearray
    """

    if not readlen:
      # force a real read request on device, but discard any result
      cmd = bytearray()
      cmd.extend((ft232h_cmds['send_immediate'],))
      self._write_data(cmd)
      self._read_data_bytes(0, 8)
      return bytearray()

    ack = (ft232h_cmds['write_bits_NVE_MSB'], 0, 0)
    nack = (ft232h_cmds['write_bits_NVE_MSB'], 0, 0xFF)

    read_not_last = ((ft232h_cmds['read_bytes_PVE_MSB'], 0, 0) + ack +
                     self._clk_lo_data_hi * self._ck_delay)
    read_last = ((ft232h_cmds['read_bytes_PVE_MSB'], 0, 0) + nack +
                 self._clk_lo_data_hi * self._ck_delay)
    # maximum RX size to fit in FTDI FIFO, minus 2 status bytes
    chunk_size = self._rx_size - 2
    cmd_size = len(read_last)
    # limit RX chunk size to the count of I2C packable commands in the FTDI
    # TX FIFO (minus one byte for the last 'send immediate' command)
    tx_count = (self._tx_size - 1) // cmd_size
    chunks = []
    if (rem := readlen) > (chunk_size := min(tx_count, chunk_size)):
      chunk_size //= 2
      cmd_chunk = bytearray()
      cmd_chunk.extend(read_not_last * chunk_size)
      cmd_chunk.extend((ft232h_cmds['send_immediate'],))

      def _write_command_gen(length: int) -> Union[bytearray, bytes]:
        if length <= 0:
          # no more data
          return b''
        if length <= chunk_size:
          cmd_ = bytearray()
          cmd_.extend(read_not_last * (length - 1))
          cmd_.extend(read_last)
          cmd_.extend((ft232h_cmds['send_immediate'],))
          return cmd_
        return cmd_chunk

      while rem:
        buf = self._read_data_bytes(rem, self._nb_attempt_1,
                                    _write_command_gen)
        chunks.append(buf)
        rem -= len(buf)
    else:
      cmd = bytearray()
      cmd.extend(read_not_last * (rem - 1))
      cmd.extend(read_last)
      cmd.extend((ft232h_cmds['send_immediate'],))
      size = rem
      self._write_data(cmd)
      buf = self._read_data_bytes(size, self._nb_attempt_2)
      chunks.append(buf)
    return bytearray(b''.join(chunks))

  def _send_check_ack(self, cmd: bytearray) -> None:
    """Actually sends the MPSSE commands generated by :meth:`_do_prolog` and
    :meth:`_do_write` methods, and checks whether the slave ACKs it.

    Args:
      cmd: The MPSSE commands to send
    """

    # SCL low, SDA high-Z
    cmd.extend(self._clk_lo_data_hi)
    # read SDA (ack from slave)
    cmd.extend((ft232h_cmds['read_bits_PVE_MSB'], 0))
    cmd.extend((ft232h_cmds['send_immediate'],))
    self._write_data(cmd)
    if not (ack := self._read_data_bytes(1, 8)):
      raise IOError('No answer from FTDI')
    if ack[0] & 0x01:
      raise IOError('NACK from slave')

  def _write_i2c(self,
                 address: int,
                 out: list,
                 stop: bool = True) -> None:
    """Writes bytes to an I2C slave.

    Args:
      address: I2C address of the slave
      out: List of bytes to send
      stop: Should the stop condition be sent at the end of the message ?
    """

    i2caddress = (address << 1) & 0xFF
    retries = self._retry_count
    while True:
      try:
        self._do_prolog(i2caddress)
        self._do_write(out)
        return
      except IOError:
        retries -= 1
        if not retries:
          raise
      finally:
        if stop:
          self._write_data(bytearray(self._stop))

  def _read_i2c(self,
                address: int,
                length: int,
                stop: bool = True) -> bytearray:
    """Reads bytes from an I2C slave.

    Args:
      address: I2C address of the slave
      length: Number of bytes to read
      stop: Should the stop condition be sent at the end of the message ?
    """

    i2caddress = (address << 1) & 0xFF
    retries = self._retry_count
    while True:
      try:
        self._do_prolog(i2caddress | 0x01)
        if len(data := self._do_read(length)) < length:
          raise IOError
        return data
      except (IOError, OSError):
        retries -= 1
        if not retries:
          raise
      finally:
        if stop:
          self._write_data(bytearray(self._stop))

  def _exchange_i2c(self,
                    address: int,
                    out: list,
                    readlen: int = 0) -> bytearray:
    """Writes bytes to an I2C slave, and then reads a given number of bytes
    from this same slave.

    Args:
      address: I2C address of the slave
      out: List of bytes to send
      readlen: Number of bytes to read

    Returns:
      Read bytes as a bytearray
    """

    if readlen < 1:
      raise IOError('Nothing to read')
    if readlen > (ft232h_max_payload / 3 - 1):
      raise IOError("Input payload is too large")
    if address is None:
      i2caddress = None
    else:
      i2caddress = (address << 1) & 0xFF
    retries = self._retry_count
    while True:
      try:
        self._do_prolog(i2caddress)
        self._do_write(out)
        self._do_prolog(i2caddress | 0x01)
        if len(data := self._do_read(readlen)) < readlen:
          raise IOError
        return data
      except (IOError, OSError):
        retries -= 1
        if not retries:
          raise
      finally:
        self._write_data(bytearray(self._stop))

  def write_byte(self, i2c_addr: int, value: int) -> None:
    """Writes a single byte to an I2C slave, in register 0.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      value: The value to write, as an :obj:`int`.
    """

    self.log(logging.DEBUG, f"Requested I2C byte write with value {value} to "
                            f"address {i2c_addr}")

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=0x00,
                              data=[value & 0xFF])

  def write_byte_data(self,
                      i2c_addr: int,
                      register: int,
                      value: int) -> None:
    """Writes a single byte to an I2C slave, in the specified register.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the register to be written, as an :obj:`int`.
      value: The value to write, as an :obj:`int`.
    """

    self.log(logging.DEBUG, f"Requested I2C byte write with value {value} to "
                            f"register {register} at address {i2c_addr}")

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[value & 0xFF])

  def write_word_data(self,
                      i2c_addr: int,
                      register: int,
                      value: int) -> None:
    """Writes 2 bytes to an I2C slave from a single int value, starting at the
    specified register.

    Depending on the size of the registers, the next register may be written as
    well.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the first register to be written, as an
        :obj:`int`.
      value: The value to write, as an :obj:`int`.
    """

    self.log(logging.DEBUG, f"Requested I2C word write with value {value} to "
                            f"register {register} at address {i2c_addr}")

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[(value >> 8) & 0xFF, value & 0xFF])

  def write_block_data(self,
                       i2c_addr: int,
                       register: int,
                       data: list) -> None:
    """Actually calls :meth:`write_i2c_block_data`.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the first register to be written, as an
        :obj:`int`.
      data: A :obj:`list` of bytes to write.
    """

    self.log(logging.DEBUG, f"Requested I2C block write with data {data} to "
                            f"register {register} at address {i2c_addr}")

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=data)

  def write_i2c_block_data(self,
                           i2c_addr: int,
                           register: int,
                           data: list) -> None:
    """Writes bytes from a :obj:`list` to an I2C slave, starting at the
    specified register.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the first register to be written, as an
        :obj:`int`.
      data: A :obj:`list` of bytes to write.
    """

    self.log(logging.DEBUG, f"Requested I2C block write with data {data} to "
                            f"register {register} at address {i2c_addr}")

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")

    self._write_i2c(address=i2c_addr,
                    out=[register] + data)

  def read_byte(self, i2c_addr: int) -> int:
    """Reads a single byte from an I2C slave, from the register `0`.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.

    Returns:
      Value of the read register
    """

    self.log(logging.DEBUG, f"Requested I2C byte read at address {i2c_addr}")

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=0x00,
                                      length=1)[0]
    except IndexError:
      self.log(logging.ERROR, "No data to read from USB device")
      raise

  def read_byte_data(self, i2c_addr: int, register: int) -> int:
    """Reads a single byte from an I2C slave, from the specified register.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the register to be read, as an :obj:`int`.

    Returns:
      Value of the read register
    """

    self.log(logging.DEBUG, f"Requested I2C byte read from register {register}"
                            f" at address {i2c_addr}")

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=register,
                                      length=1)[0]
    except IndexError:
      self.log(logging.ERROR, "No data to read from USB device")
      raise

  def read_word_data(self, i2c_addr: int, register: int) -> int:
    """Reads 2 bytes from an I2C slave, starting at the specified register, and
    returns them as one single value.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the first register to be read, as an :obj:`int`.

    Returns:
      Value of the read registers
    """

    self.log(logging.DEBUG, f"Requested I2C word read from register {register}"
                            f" at address {i2c_addr}")

    try:
      ret = self.read_i2c_block_data(i2c_addr=i2c_addr,
                                     register=register,
                                     length=2)
      return (ret[0] << 8) & ret[1]
    except IndexError:
      self.log(logging.ERROR, "Not enough data to read from USB device")
      raise

  def read_i2c_block_data(self,
                          i2c_addr: int,
                          register: int,
                          length: int) -> list[int]:
    """Reads a given number of bytes from an I2C slave, starting at the
    specified register.

    Args:
      i2c_addr: The I2C address of the slave, as an :obj:`int`.
      register: The index of the first register to be read, as an :obj:`int`.
      length: The number of bytes to read, as an :obj:`int`.

    Returns:
      Values of read registers as a :obj:`list`
    """

    self.log(logging.DEBUG, f"Requested I2C block read of length {length} from"
                            f" register {register} at address {i2c_addr}")

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")
    if not length >= 0:
      raise ValueError("length should be a positive integer")

    if length == 0:
      return []
    retries = 2
    while True:
      try:
        ret = [byte for byte in self._exchange_i2c(address=i2c_addr,
                                                   out=[register],
                                                   readlen=length)]
        return ret
      except (IOError, OSError):
        retries -= 1
        if not retries:
          raise

  def i2c_rdwr(self, *i2c_msgs: I2CMessage) -> None:
    """Exchanges messages with a slave that doesn't feature registers.

    A start condition is sent at the beginning of each transaction, but only
    one stop condition is sent after the last transaction.

    Args:
      *i2c_msgs: One or several :class:`~crappy.tool.ft232h.I2CMessage` to
        exchange with the slave. They are either read or write messages.
    """

    self.log(logging.DEBUG, "Requested I2C readwrite")

    nr = len(i2c_msgs)
    for i, msg in enumerate(i2c_msgs):
      if msg.type == 'w':
        self._write_i2c(address=msg.addr, out=msg.buf, stop=(i == nr))
      elif msg.type == 'r':
        msg.buf = [byte for byte in self._read_i2c(address=msg.addr,
                                                   length=msg.len,
                                                   stop=(i == nr))]

  @property
  def bits_per_word(self) -> int:
    """Number of bits per SPI words.

    Can only be set to 8.
    """

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._bits_per_word

  @bits_per_word.setter
  def bits_per_word(self, value: int) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, int):
      raise TypeError("bits_per_word should be an integer")
    if value != 8:
      raise ValueError("bits_per_word values other than 8 are not implemented")

    self.log(logging.DEBUG, f"Set SPI bits_per_word to {value}")
    self._bits_per_word = value

  @property
  def cshigh(self) -> bool:
    """If :obj:`True`, the polarity of the CS line is inverted."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._cshigh

  @cshigh.setter
  def cshigh(self, value: bool) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("cshigh should be either True or False")
    self._spi_param_changed = True

    self.log(logging.DEBUG, f"Set SPI cshigh to {value}")
    self._cshigh = value

  @property
  def loop(self) -> bool:
    """If :obj:`True`, the loopback mode is enabled."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._loop

  @loop.setter
  def loop(self, value: bool) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("loop should be either True or False")
    if value:
      self._write_data(bytearray((ft232h_cmds['loopback_start'],)))
    else:
      self._write_data(bytearray((ft232h_cmds['loopback_end'],)))

    self.log(logging.DEBUG, f"Set SPI loop to {value}")
    self._loop = value

  @property
  def no_cs(self) -> bool:
    """If :obj:`True`, the CS line is not driven."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._no_cs

  @no_cs.setter
  def no_cs(self, value: bool) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("no_cs should be either True or False")

    self.log(logging.DEBUG, f"Set SPI no_cs to {value}")
    self._no_cs = value

  @property
  def lsbfirst(self) -> bool:
    """If :obj:`True`, data is sent and received LSB first."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._lsbfirst

  @lsbfirst.setter
  def lsbfirst(self, value: bool) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("lsbfirst should be either True or False")

    self.log(logging.DEBUG, f"Set SPI lsbfirst to {value}")
    self._lsbfirst = value

  @property
  def max_speed_hz(self) -> float:
    """The SPI bus clock frequency in Hz.

    In SPI modes `1` and `3`, the actual bus clock frequency is 50% higher than
    the input value because the FT232H is switched to 3-phase clock mode.
    """

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._max_speed_hz

  @max_speed_hz.setter
  def max_speed_hz(self, value: float) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if self.mode in [1, 3]:
      if not 3200 <= value <= 2 * ft232h_clock['high'] / 3:
        raise ValueError("max_speed_hz should be between 3.2kHz and 20MHz in "
                         "spi modes 1 and 3")
    else:
      if not 4800 <= value <= ft232h_clock['high']:
        raise ValueError("max_speed_hz should be between 4.8kHz and 30MHz in "
                         "spi modes 0 and 2")
    self._spi_param_changed = True
    if self.mode in [1, 3]:
      self._set_frequency(3 * value / 2)
      self._write_data(bytearray([True and
                                  ft232h_cmds['enable_clk_3phase'] or
                                  ft232h_cmds['disable_clk_3phase']]))
    else:
      self._set_frequency(value)
      self._write_data(bytearray([False and
                                  ft232h_cmds['enable_clk_3phase'] or
                                  ft232h_cmds['disable_clk_3phase']]))

    self.log(logging.DEBUG, f"Set SPI max_speed_hz to {value}")
    self._max_speed_hz = value

  @property
  def mode(self) -> int:
    """The SPI mode used for communicating with the slave.

    When changing the SPI mode, the bus clock frequency may be reloaded.
    """

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._mode

  @mode.setter
  def mode(self, value: int) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if value not in range(4):
      raise ValueError("mode should be an integer between 0 and 3")
    former_mode = self.mode

    self.log(logging.DEBUG, f"Set SPI mode to {value}")
    self._mode = value
    self._spi_param_changed = True
    if value % 2 != former_mode % 2:
      self.max_speed_hz = self.max_speed_hz

  @property
  def threewire(self) -> bool:
    """If :obj:`True`, indicates that the MISO and MOSI lines are connected
    together. Not currently implemented."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._threewire

  @threewire.setter
  def threewire(self, value: bool) -> None:
    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("threewire should be either True or False")
    if value:
      raise ValueError("threewire mode not implemented")

    self.log(logging.DEBUG, f"Set SPI threewire to {value}")
    self._threewire = value

  def _exchange_spi(self,
                    readlen: int,
                    out: list,
                    start: bool,
                    stop: bool,
                    duplex: bool) -> bytes:
    """Exchanges bytes with an SPI slave.

    Can read and/or write data, in a sequential or simultaneous way. Also
    manages the CS line.

    Args:
      readlen: The umber of bytes to read, as an :obj:`int`. If 0, no reading
        is performed.
      out: A :obj:`list` of bytes to write. If empty, no writing is performed.
      start: If :obj:`False`, the CS line is not driven before exchanging data,
        and remains in its previous state.
      stop: If :obj:`False`, the CS line is not driven after exchanging data,
        and remains in its previous state.
      duplex: If :obj:`True`, the data is read and written simultaneously. If
        :obj:`False`, writes then reads in a sequential way.

    Returns:
      Read data as bytes
    """

    if len(out) > ft232h_max_payload:
      raise IOError("Output payload is too large")
    if readlen > ft232h_max_payload:
      raise IOError("Input payload is too large")

    if not isinstance(out, bytearray):
      out = bytearray(out)

    # Re-building the _cs_prolog and _cs_epilog if an SPI parameter has been
    # modified
    if self._spi_param_changed:
      cs_hold = 1 + int(1E6 / self.max_speed_hz)
      self._cpol = self.mode & 0x2
      cs_clock = 0xFFFF & ~((~self._cpol & ft232h_pins['SCK']) |
                            ft232h_pins['DO'] |
                            (self.cshigh and self._cs_bit))
      cs_select = 0xFFFF & ~((~self._cpol & ft232h_pins['SCK']) |
                             ft232h_pins['DO'] |
                             ((not self.cshigh) and self._cs_bit))
      self._cs_prolog = [cs_clock, cs_select]
      self._cs_epilog = [cs_select] + [cs_clock] * cs_hold

      self._spi_param_changed = False

    # Building the prolog MPSSE command
    cmd = bytearray()
    if start:
      for ctrl in self._cs_prolog:
        ctrl &= self._spi_mask
        ctrl |= self._gpio_low
        ctrl |= self._gpio_high << 8
        cmd.extend((ft232h_cmds['set_bits_low'], ctrl & 0xFF,
                    self._direction & 0xFF))

    # Building the epilog MPSSE command
    epilog = bytearray()
    for ctrl in self._cs_epilog:
      ctrl &= self._spi_mask
      ctrl |= self._gpio_low
      ctrl |= self._gpio_high << 8
      epilog.extend((ft232h_cmds['set_bits_low'], ctrl & 0xFF,
                     self._direction & 0xFF))

    # Restore idle state
    if not self.cshigh:
      cs_high = [ft232h_cmds['set_bits_low'], self._cs_bit |
                 self._gpio_low & 0xFF,
                 self._direction & 0xFF]
    else:
      cs_high = [ft232h_cmds['set_bits_low'], self._gpio_low & 0xFF,
                 self._direction & 0xFF]

    if not self._turbo:
      cs_high.append(ft232h_cmds['send_immediate'])
    epilog.extend(cs_high)

    # Sequential communication
    if not duplex:
      # Write MPSSE commands
      writelen = len(out)
      if writelen:
        if not self.lsbfirst:
          wcmd = (ft232h_cmds['write_bytes_NVE_MSB'] if not self._cpol else
                  ft232h_cmds['write_bytes_PVE_MSB'])
        else:
          wcmd = (ft232h_cmds['write_bytes_NVE_LSB'] if not self._cpol else
                  ft232h_cmds['write_bytes_PVE_LSB'])
        write_cmd = pack('<BH', wcmd, writelen - 1)
        cmd.extend(write_cmd)
        cmd.extend(out)

      # Read MPSSE commands
      if readlen:
        if not self.lsbfirst:
          rcmd = (ft232h_cmds['read_bytes_NVE_MSB'] if not self._cpol else
                  ft232h_cmds['read_bytes_PVE_MSB'])
        else:
          rcmd = (ft232h_cmds['read_bytes_NVE_LSB'] if not self._cpol else
                  ft232h_cmds['read_bytes_PVE_LSB'])
        read_cmd = pack('<BH', rcmd, readlen - 1)
        cmd.extend(read_cmd)
        # ====================================================================
        if self._turbo:
          cmd.extend((ft232h_cmds['send_immediate'],))

        if self._turbo:
          if stop:
            cmd.extend(epilog)
          self._write_data(cmd)
        else:
          self._write_data(cmd)
          if stop:
            self._write_data(epilog)
        # USB read cycle may occur before the FTDI device has actually
        # sent the data, so try to read more than once if no data is
        # actually received
        data = self._read_data_bytes(readlen, 8)

      # If nothing to read
      else:
        if writelen:
          if self._turbo:
            if stop:
              cmd.extend(epilog)
            self._write_data(cmd)
          else:
            self._write_data(cmd)
            if stop:
              self._write_data(epilog)
        data = bytearray()

    # Simultaneous communication
    else:
      if readlen > len(out):
        tmp = bytearray(out)
        tmp.extend([0] * (readlen - len(out)))
        out = tmp

      exlen = len(out)
      if not self.lsbfirst:
        wcmd = (ft232h_cmds['rw_bytes_PVE_NVE_MSB'] if not self._cpol else
                ft232h_cmds['rw_bytes_NVE_PVE_MSB'])
      else:
        wcmd = (ft232h_cmds['rw_bytes_PVE_NVE_LSB'] if not self._cpol else
                ft232h_cmds['rw_bytes_NVE_PVE_LSB'])
      write_cmd = pack('<BH', wcmd, exlen - 1)
      cmd.extend(write_cmd)
      cmd.extend(out)
      # ======================================================================
      if self._turbo:
        cmd.extend((ft232h_cmds['send_immediate'],))

      if self._turbo:
        if stop:
          cmd.extend(epilog)
        self._write_data(cmd)
      else:
        self._write_data(cmd)
        if stop:
          self._write_data(epilog)
      # USB read cycle may occur before the FTDI device has actually
      # sent the data, so try to read more than once if no data is
      # actually received
      data = self._read_data_bytes(exlen, 8)
    return data

  def readbytes(self,
                len: int,
                start: bool = True,
                stop: bool = True) -> list[int]:
    """Reads the specified number of bytes from an SPI slave.

    Args:
      len: The number of bytes to read, as an :obj:`int`.
      start: If :obj:`False`, the CS line is not driven before reading data,
        and remains in its previous state.
      stop: If :obj:`False`, the CS line is not driven after reading data, and
        remains in its previous state.

    Returns:
      List of read bytes
    """

    self.log(logging.DEBUG, f"Requested SPI bytes read of length {len}")

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    return [byte for byte in self._exchange_spi(readlen=len,
                                                out=[],
                                                start=start,
                                                stop=stop,
                                                duplex=False)]

  def writebytes(self,
                 values: list,
                 start: bool = True,
                 stop: bool = True) -> None:
    """Write bytes from a list to an SPI slave.

    Args:
      values: A :obj:list` of bytes to write
      start: If :obj:`False`, the CS line is not driven before reading data,
        and remains in its previous state.
      stop: If :obj:`False`, the CS line is not driven after reading data, and
        remains in its previous state.
    """

    self.log(logging.DEBUG, f"Requested SPI bytes write with values {values}")

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    self._exchange_spi(readlen=0,
                       out=values,
                       start=start,
                       stop=stop,
                       duplex=False)

  def writebytes2(self,
                  values: list,
                  start: bool = True,
                  stop: bool = True) -> None:
    """Actually calls the :meth:`writebytes` method with the same arguments."""

    self.log(logging.DEBUG, f"Requested SPI bytes write with values {values}")

    self.writebytes(values=values,
                    start=start,
                    stop=stop)

  def xfer(self,
           values: list,
           speed: Optional[float] = None,
           delay: float = 0.0,
           bits: int = 8,
           start: bool = True,
           stop: bool = True) -> list[int]:
    """Simultaneously reads and write bytes to an SPI slave.

    The number of bytes to read is equal to the number of bytes in the write
    buffer.

    Args:
      values: A :obj:list` of bytes to write.
      speed: Sets the bus clock frequency in Hz before issuing the command, as
        a :obj:`float`.
      delay: Not implemented, should be 0.0
      bits:  Not implemented, should be 8
      start: If :obj:`False`, the CS line is not driven before reading data,
        and remains in its previous state.
      stop: If :obj:`False`, the CS line is not driven after reading data, and
        remains in its previous state.

    Returns:
      List of read bytes
    """

    self.log(logging.DEBUG, f"Requested SPI xfer with values {values}")

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    if bits != 8:
      raise ValueError("Only bits=8 is implemented")
    if delay != 0.0:
      raise ValueError("The delay parameter is not currently implemented")

    if speed != self.max_speed_hz and speed is not None:
      self.max_speed_hz = speed

    return [byte for byte in self._exchange_spi(readlen=len(values),
                                                out=values,
                                                start=start,
                                                stop=stop,
                                                duplex=True)]

  def xfer2(self,
            values: list,
            speed: float = 6.0E6,
            delay: float = 0.0,
            bits: int = 8,
            start: bool = True,
            stop: bool = True) -> list[int]:
    """Actually calls the :meth:`xfer` method with the same arguments."""

    self.log(logging.DEBUG, f"Requested SPI xfer with values {values}")

    return self.xfer(values=values,
                     speed=speed,
                     delay=delay,
                     bits=bits,
                     start=start,
                     stop=stop)

  def xfer3(self,
            values: list,
            speed: float = 6.0E6,
            delay: float = 0.0,
            bits: int = 8,
            start: bool = True,
            stop: bool = True) -> list[int]:
    """Actually calls the :meth:`xfer` method with the same arguments."""

    self.log(logging.DEBUG, f"Requested SPI xfer with values {values}")

    return self.xfer(values=values,
                     speed=speed,
                     delay=delay,
                     bits=bits,
                     start=start,
                     stop=stop)

  @property
  def _gpio_all_pins(self) -> int:
    """Reports the addressable GPIOs as a bitfield.

    A :obj:`True` bit represents a pin which may be used as a GPIO, a
    :obj:`False` bit a reserved pin.

    Returns:
      Bitfield of configurable GPIO pins
    """

    mask = (1 << ft232h_port_width) - 1
    if self._ft232h_mode == 'I2C':
      return mask & ~self._i2c_mask
    elif self._ft232h_mode == 'SPI':
      return mask & ~self._spi_mask
    else:
      return mask

  @property
  def _direction(self) -> int:
    """Provides the FTDI pin direction.

    A :obj:`True` bit represents an output pin, a :obj:`False` bit an input
    pin.

    Returns:
      Bitfield of pins direction.
    """

    if self._ft232h_mode == 'I2C':
      return self._i2c_dir | self._gpio_dir
    elif self._ft232h_mode == 'SPI':
      no_cs_mask = 0xFFFF - (self._cs_bit if self.no_cs else 0)
      return self._spi_dir | self._gpio_dir & no_cs_mask
    else:
      return self._gpio_dir

  def _read_gpio_raw(self) -> int:
    """Sends the MPSSE commands for reading all the FT232H pins, and returns
    the bitmap of read values. Values are determined using 3.3V logic.

    Returns:
      Bitmap of pins values
    """

    cmd = bytes([ft232h_cmds['get_bits_low'],
                 ft232h_cmds['get_bits_high'],
                 ft232h_cmds['send_immediate']])
    fmt = '<H'
    self._write_data(cmd)
    size = calcsize(fmt)
    data = self._read_data_bytes(size, 8)
    if len(data) != size:
      raise IOError('Cannot read GPIO')
    value, = unpack(fmt, data)
    return value

  def get_gpio(self, gpio_str: str) -> bool:
    """Reads the 3.3V-logic voltage value of the specified pin.

    Args:
      gpio_str: The name of the GPIO to be read, as a :obj:`str`.

    Returns:
      3.3V-logic value corresponding to the input voltage
    """

    self.log(logging.DEBUG, f"Requested GPIO value reading for {gpio_str}")

    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    if not self._gpio_all_pins & (gpio_bit := ft232h_pin_nr[gpio_str]):
      raise ValueError("Cannot use pin {} as a GPIO".format(gpio_str))

    # Changing the _direction and _gpio_dir bitfields
    if self._direction & gpio_bit:
      self._gpio_dir &= 0xFFFF - gpio_bit

    return bool(self._read_gpio_raw() & gpio_bit)

  def set_gpio(self, gpio_str: str, value: int) -> None:
    """Sets the specified GPIO as an output and sets its output value.

    Args:
      gpio_str: The name of the GPIO to be set, as a :obj:`str`.
      value: 1 for setting the GPIO high, 0 for setting it low.
    """

    self.log(logging.DEBUG, f"Requested GPIO value writing to {value} "
                            f"for {gpio_str}")

    if value not in [0, 1]:
      raise ValueError("value should be either 0 or 1")
    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    if not self._gpio_all_pins & (gpio_bit := ft232h_pin_nr[gpio_str]):
      raise ValueError("Cannot use pin {} as a GPIO".format(gpio_str))

    # Changing the _direction and _gpio_dir bitfields
    if not (self._direction & gpio_bit):
      self._gpio_dir |= gpio_bit

    data = self._read_gpio_raw()
    if value == 1:
      data |= gpio_bit
    else:
      data &= 0xFFFF - gpio_bit
    low_data = data & 0xFF
    low_dir = self._direction & 0xFF
    high_data = (data >> 8) & 0xFF
    high_dir = (self._direction >> 8) & 0xFF
    cmd = bytes([ft232h_cmds['set_bits_low'], low_data, low_dir,
                 ft232h_cmds['set_bits_high'], high_data, high_dir])
    self._write_data(cmd)
    self._gpio_low = low_data & self._gpio_all_pins
    self._gpio_high = high_data & self._gpio_all_pins

  def close(self) -> None:
    """Closes the FTDI interface/port."""

    if self._usb_dev:
      self.log(logging.INFO, "Closing the USB connection to the FT232H")
      if bool(self._usb_dev._ctx.handle):
        try:
          self._set_bitmode(0, FT232H.BitMode.RESET)
          util.release_interface(self._usb_dev, self._index - 1)
        except (IOError, ValueError, USBError):
          pass
        try:
          self._usb_dev.attach_kernel_driver(self._index - 1)
        except (NotImplementedError, USBError):
          pass
      self.log(logging.INFO, "Releasing the USB resources")
      util.dispose_resources(self._usb_dev)
      self._usb_dev = None
