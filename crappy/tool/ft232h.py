# coding: utf-8

from enum import IntEnum
from collections import namedtuple
from struct import calcsize, unpack, pack
from typing import Union, Any
from collections.abc import Callable
from multiprocessing import Queue, Event
from multiprocessing.managers import Namespace
from time import time

from .._global import OptionalModule
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

# Todo:
# Check Windows compatibility (is kernel driver active)
# Add fast FT232H mode ?

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


class Find_serial_number:
  """A class used for finding USB devices matching a given serial number, using
     the usb.core.find method."""

  def __init__(self, serial_number: str) -> None:
    self.serial_number = serial_number

  def __call__(self, device) -> bool:
    return device.serial_number == self.serial_number


class ft232h:
  """A class for controlling FTDI's USB to Serial FT232H.

  Communication in SPI and I2C are implemented, along with GPIO control. The
  name of the methods for SPI and I2C communication are those of :mod:`smbus`
  and :mod:`spidev` libraries, in order to facilitate the use and the
  integration in a multi-backend environment. This class also allows to write a
  USB serial number in the EEPROM, as there's no default serial number on the
  chip.

  Important:
    If using Adafruit's board, its `I2C Mode` switch should of course be set to
    the correct value according to the chosen mode.

  Important:
    **Only for Linux users:** In order to drive the FT232H, the appropriate
    udev rule should be set. This can be done using the `udev_rule_setter`
    utility in ``crappy``'s `util` folder. It is also possible to add it
    manually by running:
    ::

      $ echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"0403\\", \
MODE=\\"0666\\\"" | sudo tee ftdi.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.

  Important:
    For controlling several FT232H from the same computer, it is first necessary
    to set their USB serial numbers. Otherwise an error will be raised. This can
    be done using the crappy utility ``Set_ft232h_serial_nr.py``.
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
               mode: str,
               serial_nr: str = None,
               i2c_speed: float = 100E3,
               spi_turbo: bool = False) -> None:
    """Checks the arguments validity, initializes the device and sets the locks.

    Args:
      mode (:obj:`str`): The communication mode, can be :
        ::

          'SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'

        GPIOs can be driven in any mode, but faster speeds are achievable in
        `GPIO_only` mode.
      serial_nr (:obj:`str`, optional): The serial number of the FT232H to
        drive. In `Write_serial_nr` mode, the serial number to be written.
      i2c_speed (:obj:`str`, optional): In I2C mode, the I2C bus clock frequency
        in Hz. Available values are :
        ::

          100E3, 400E3, 1E6.

      spi_turbo (:obj:`str`, optional): Increases the achievable bus speed, but
        may not work with some devices.

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
      raise ValueError("i2c_speed should be in {}".format(list(
        ft232h_i2c_speed.values())))

    self._gpio_low = 0
    self._gpio_high = 0
    self._gpio_dir = 0
    self._retry_count = 8

    self._usb_write_timeout = 5000
    self._usb_read_timeout = 5000

    self._serial_nr = serial_nr
    self._turbo = spi_turbo
    self._i2c_speed = i2c_speed

    self._initialize()

    if mode == 'Write_serial_nr':
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
      timings = ft232h_i2c_speed[self._i2c_speed]
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

      self._bits_per_word = 8
      self._cshigh = False
      self._no_cs = False
      self._loop = False
      self._lsbfirst = False
      self._max_speed_hz = 400E3
      self._mode = 0
      self._threewire = False
      self._spi_param_changed = True

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
                     custom_match=Find_serial_number(
                       self._serial_nr))
    else:
      devices = find(find_all=True,
                     idVendor=Ftdi_vendor_id,
                     idProduct=ft232h_product_id)

    # Checking if there's only 1 device matching
    devices = list(devices)
    if len(devices) == 0:
      raise IOError("No matching ft232h connected")
    elif len(devices) > 1:
      raise IOError("Several ft232h devices found, please specify a serial_nr")
    else:
      self._usb_dev = devices[0]

    try:
      self._serial_nr = self._usb_dev.serial_number
    except ValueError:
      self._serial_nr = ""

    # Configuring the USB device, interface and endpoints
    try:
      if self._usb_dev.is_kernel_driver_active(0):
        self._usb_dev.detach_kernel_driver(0)
      self._usb_dev.set_configuration()
    except USBError:
      print("You may have to install the udev-rules for this USB device, "
            "this can be done using the udev_rule_setter utility in the util "
            "folder")
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
    self._set_bitmode(0, ft232h.BitMode.RESET)

    # Set latency timer
    self._set_latency_timer(latency)

    # Set chunk size and invalidate all remaining data
    self._writebuffer_chunksize = fifo_sizes[0]
    self._readoffset = 0
    self._readbuffer = bytearray()
    self._readbuffer_chunksize = min(fifo_sizes[0], fifo_sizes[1],
                                     self._max_packet_size)

    # Reset feature mode
    self._set_bitmode(0, ft232h.BitMode.RESET)
    # Drain buffers
    self._purge_buffers()
    # Disable event and error characters
    if self._ctrl_transfer_out(ft232h_sio_req['set_event_char'], 0):
      raise IOError('Unable to set event char')
    if self._ctrl_transfer_out(ft232h_sio_req['set_error_char'], 0):
      raise IOError('Unable to set error char')

    # Enable MPSSE mode
    if self._ft232h_mode == 'GPIO_only':
      self._set_bitmode(0xFF, ft232h.BitMode.BITBANG)
    else:
      self._set_bitmode(self._direction, ft232h.BitMode.MPSSE)

    # Configure clock
    if self._ft232h_mode == 'I2C':
      # Note that bus frequency may differ from clock frequency, when
      # 3-phase clock is enabled
      self._set_frequency(3 * frequency / 2)
    else:
      self._set_frequency(frequency)

    # Configure pins
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
    self._write_data(bytearray((ft232h_cmds['loopback_end'],)))
    # Validate MPSSE
    bytes_ = bytes(self._read_data_bytes(2))
    if (len(bytes_) >= 2) and (bytes_[0] == '\xfa'):
      raise IOError("Invalid command @ %d" % bytes_[1])

    # I2C-specific settings
    if self._ft232h_mode == 'I2C':
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

  @staticmethod
  def _compute_delay_cycles(value: float) -> int:
    """Approximates the number of clock cycles over a given delay.

    Args:
      value (:obj:`float`): delay (in seconds)

    Returns:
      Number of clock cycles
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
      latency (:obj:`int`): latency (in milliseconds)
    """

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
      frequency (:obj:`float`): Desired bus frequency (in Hz)

    Returns:
      Actual bus frequency
    """

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
      bitmask (:obj:`int`): Mask for choosing the driven GPIOs.
      mode (:class:`BitMode`): Bitbang mode to be set.
    """

    mask = sum(ft232h.BitMode)
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
      reqtype (:obj:`int`): bmRequest
      value (:obj:`int`): wValue
      data (:obj:`bytes`): payload

    Returns:
      Number of bytes actually written
    """

    try:
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
      serial_number (:obj:`str`): Serial number to be written in the EEPROM
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
        buf = self._usb_dev.ctrl_transfer(
          Ftdi_req_in, ft232h_sio_req['read_eeprom'], 0,
          word_addr, 2, self._usb_read_timeout)
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
      print("An error occurred while writing to USB")
      raise

  def _read_data_bytes(self,
                       size: int,
                       attempt: int = 2,
                       request_gen: Callable = None) -> bytes:
    """Reads data from the FT232H.

    Reads data from the FTDI interface. The data buffer is rebuilt from
    chunk-sized blocks received over the USB bus. The FTDI device always sends
    internal status bytes, which are stripped out as not part of the data
    payload.

    Args:
      size (:obj:`int`): The number of bytes to receive from the device
      attempt (:obj:`int`): Attempt cycle count
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
            tempbuf = self._usb_dev.read(self._out_ep,
                                         self._readbuffer_chunksize,
                                         self._usb_read_timeout)
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
                req_size -= length - 2
                if req_size > 0:
                  cmd = request_gen(req_size)
                  if cmd:
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
      print("An error occurred while writing to USB")
      raise
    # never reached
    raise ValueError("Internal error")

  @property
  def _clk_hi_data_lo(self) -> tuple:
    """Returns the MPSSE command for driving CLK line high and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SCL'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_input(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and listening to SDA
       line, while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            ft232h_pins['SCL'] | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_hi(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SDAO'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_lo(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _idle(self) -> tuple:
    """Returns the MPSSE command for driving CLK line high and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._i2c_dir | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _start(self) -> tuple:
    """Returns the MPSSE command for issuing and I2C start condition."""

    return self._clk_hi_data_lo * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta

  @property
  def _stop(self) -> tuple:
    """Returns the MPSSE command for issuing and I2C stop condition."""

    return self._clk_lo_data_hi * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta + \
        self._clk_hi_data_lo * self._ck_su_sto + \
        self._idle * self._ck_idle

  def _do_prolog(self, i2caddress: int) -> None:
    """Sends the MPSSE commands for starting an I2C transaction.

    Args:
      i2caddress (:obj:`int`): I2C address of the slave
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
      out (:obj:`list`): List of bytes to write
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
    """
    Sends the MPSSE commands for reading bytes from an I2C slave, and then
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
    chunk_size = min(tx_count, chunk_size)
    chunks = []
    rem = readlen
    if rem > chunk_size:
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
        buf = self._read_data_bytes(rem, 8, _write_command_gen)
        chunks.append(buf)
        rem -= len(buf)
    else:
      cmd = bytearray()
      cmd.extend(read_not_last * (rem - 1))
      cmd.extend(read_last)
      cmd.extend((ft232h_cmds['send_immediate'],))
      size = rem
      self._write_data(cmd)
      buf = self._read_data_bytes(size, 8)
      chunks.append(buf)
    return bytearray(b''.join(chunks))

  def _send_check_ack(self, cmd: bytearray) -> None:
    """Actually sends the MPSSE commands generated by :meth:`_do_prolog` and
    :meth:`_do_write` methods, and checks whether the slave ACKs it.

    Args:
      cmd (:obj:`bytearray`): The MPSSE commands to send
    """

    # SCL low, SDA high-Z
    cmd.extend(self._clk_lo_data_hi)
    # read SDA (ack from slave)
    cmd.extend((ft232h_cmds['read_bits_PVE_MSB'], 0))
    cmd.extend((ft232h_cmds['send_immediate'],))
    self._write_data(cmd)
    ack = self._read_data_bytes(1, 8)
    if not ack:
      raise IOError('No answer from FTDI')
    if ack[0] & 0x01:
      raise IOError('NACK from slave')

  def _write_i2c(self, address: int, out: list) -> None:
    """Writes bytes to an I2C slave.

    Args:
      address (:obj:`int`): I2C address of the slave
      out (:obj:`list`): List of bytes to send
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
        self._write_data(bytearray(self._stop))

  def _exchange_i2c(self,
                    address: int,
                    out: list,
                    readlen: int = 0) -> bytearray:
    """Writes bytes to an I2C slave, and then reads a given number of bytes
    from this same slave.

    Args:
      address (:obj:`int`): I2C address of the slave
      out (:obj:`list`): List of bytes to send
      readlen (:obj:`int`): Number of bytes to read

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
        data = self._do_read(readlen)
        return data
      except IOError:
        retries -= 1
        if not retries:
          raise
      finally:
        self._write_data(bytearray(self._stop))

  def write_byte(self, i2c_addr: int, value: int) -> None:
    """Writes a single byte to an I2C slave, in register 0.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=0x00,
                              data=[value & 0xFF])

  def write_byte_data(self, i2c_addr: int, register: int, value: int) -> None:
    """Writes a single byte to an I2C slave, in the specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the register to be written
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[value & 0xFF])

  def write_word_data(self, i2c_addr: int, register: int, value: int) -> None:
    """Writes 2 bytes to an I2C slave from a single int value, starting at the
    specified register.

    Depending on the size of the registers, the next register may be written as
    well.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[(value >> 8) & 0xFF, value & 0xFF])

  def write_block_data(self, i2c_addr: int, register: int, data: list) -> None:
    """Actually calls :meth:`write_i2c_block_data`.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      data (:obj:`list`): List of bytes to write
    """

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
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      data (:obj:`list`): List of bytes to write
    """

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")

    self._write_i2c(address=i2c_addr,
                    out=[register] + data)

  def read_byte(self, i2c_addr: int) -> int:
    """Reads a single byte from an I2C slave, from the register `0`.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave

    Returns:
      Value of the read register
    """

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=0x00,
                                      length=1)[0]
    except IndexError:
      print("No data to read")
      raise

  def read_byte_data(self, i2c_addr: int, register: int) -> int:
    """Reads a single byte from an I2C slave, from the specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the register to be read

    Returns:
      Value of the read register
    """

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=register,
                                      length=1)[0]
    except IndexError:
      print("No data to read")
      raise

  def read_word_data(self, i2c_addr: int, register: int) -> int:
    """Reads 2 bytes from an I2C slave, starting at the specified register, and
    returns them as one single value.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be read

    Returns:
      Value of the read registers
    """

    try:
      ret = self.read_i2c_block_data(i2c_addr=i2c_addr,
                                     register=register,
                                     length=2)
      return (ret[0] << 8) & ret[1]
    except IndexError:
      print("Not enough data to read")
      raise

  def read_i2c_block_data(self,
                          i2c_addr: int,
                          register: int,
                          length: int) -> list:
    """Reads a given number of bytes from an I2C slave, starting at the
    specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be read
      length (:obj:`int`): Number of bytes to read

    Returns:
      Values of read registers as a :obj:`list`
    """

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")
    if not length >= 0:
      raise ValueError("length should be a positive integer")

    if length == 0:
      return []
    return [byte for byte in self._exchange_i2c(address=i2c_addr,
                                                out=[register],
                                                readlen=length)]

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
    self._threewire = value

  def _exchange_spi(self, readlen: int, out: list, start: bool,
                    stop: bool, duplex: bool) -> bytes:
    """Exchanges bytes with an SPI slave.

    Can read and/or write data, in a sequential or simultaneous way. Also
    manages the CS line.

    Args:
      readlen (:obj:`int`): Number of bytes to read. If 0, no reading is
        performed.
      out (:obj:`list`): List of bytes to write. If empty, no writing is
        performed.
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        exchanging data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        exchanging data, and remains in its previous state.
      duplex (:obj:`int`): If :obj:`True`, the data is read and written
        simultaneously. If :obj:`False`, writes then reads in a sequential way.

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

  def readbytes(self, len: int, start: bool = True, stop: bool = True) -> list:
    """Reads the specified number of bytes from an SPI slave.

    Args:
      len (:obj:`int`): Number of bytes to read
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.

    Returns:
      List of read bytes
    """

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
      values (:obj:`list`): List of bytes to write
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.
    """

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

    self.writebytes(values=values,
                    start=start,
                    stop=stop)

  def xfer(self,
           values: list,
           speed: float = None,
           delay: float = 0.0,
           bits: int = 8,
           start: bool = True,
           stop: bool = True) -> list:
    """Simultaneously reads and write bytes to an SPI slave.

    The number of bytes to read is equal to the number of bytes in the write
    buffer.

    Args:
      values (:obj:`list`): List of bytes to write
      speed (:obj:`float`): Sets the bus clock frequency before issuing the
        command (in Hz)
      delay (:obj:`float`): Not implemented, should be 0.0
      bits (:obj:`int`):  Not implemented, should be 8
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.

    Returns:
      List of read bytes
    """

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
            stop: bool = True) -> list:
    """Actually calls the :meth:`xfer` method with the same arguments."""

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
            stop: bool = True) -> list:
    """Actually calls the :meth:`xfer` method with the same arguments."""

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

    A :obj:`True` bit represents an output pin, a :obj:`False` bit an input pin.

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
    """Sends the MPSSE commands for reading all the FT232H pins, and returns the
    bitmap of read values. Values are determined using 3.3V logic.

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

  def get_gpio(self, gpio_str: str) -> int:
    """Reads the 3.3V-logic voltage value of the specified pin.

    Args:
      gpio_str (:obj:`str`): Name of the GPIO to be read

    Returns:
      3.3V-logic value corresponding to the input voltage
    """

    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    gpio_bit = ft232h_pin_nr[gpio_str]
    if not self._gpio_all_pins & gpio_bit:
      raise ValueError("Cannot use pin {} as a GPIO".format(gpio_str))

    # Changing the _direction and _gpio_dir bitfields
    if self._direction & gpio_bit:
      self._gpio_dir &= 0xFFFF - gpio_bit

    return self._read_gpio_raw() & gpio_bit

  def set_gpio(self, gpio_str: str, value: int) -> None:
    """Sets the specified GPIO as an output and sets its output value.

    Args:
      gpio_str (:obj:`str`): Name of the GPIO to be set
      value (:obj:`int`): 1 for setting the GPIO high, 0 for setting it low
    """

    if value not in [0, 1]:
      raise ValueError("value should be either 0 or 1")
    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    gpio_bit = ft232h_pin_nr[gpio_str]
    if not self._gpio_all_pins & gpio_bit:
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
      if bool(self._usb_dev._ctx.handle):
        try:
          self._set_bitmode(0, ft232h.BitMode.RESET)
          util.release_interface(self._usb_dev, self._index - 1)
        except (IOError, ValueError, USBError):
          pass
        try:
          self._usb_dev.attach_kernel_driver(self._index - 1)
        except (NotImplementedError, USBError):
          pass
      util.dispose_resources(self._usb_dev)
      self._usb_dev = None


class ft232h_server:
  """A class for controlling FTDI's USB to Serial FT232H.

  This class is very similar to the :class:`ft232h` except it doesn't
  directly instantiate the USB device nor send commands to it directly. Instead
  the commands are sent to a USB server managing communication with the
  different FT232H devices.

  Communication in SPI and I2C are implemented, along with GPIO control. The
  name of the methods for SPI and I2C communication are those of :mod:`smbus`
  and :mod:`spidev` libraries, in order to facilitate the use and the
  integration in a multi-backend environment. This class also allows to write a
  USB serial number in the EEPROM, as there's no default serial number on the
  chip.

  Important:
    If using Adafruit's board, its `I2C Mode` switch should of course be set to
    the correct value according to the chosen mode.

  Important:
    **Only for Linux users:** In order to drive the FT232H, the appropriate
    udev rule should be set. This can be done using the `udev_rule_setter`
    utility in ``crappy``'s `util` folder. It is also possible to add it
    manually by running:
    ::

      $ echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"0403\\", \
MODE=\\"0666\\\"" | sudo tee ftdi.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.

  Important:
    For controlling several FT232H from the same computer, it is first necessary
    to set their USB serial numbers. Otherwise an error will be raised. This can
    be done using the crappy utility ``Set_ft232h_serial_nr.py``.
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
               mode: str,
               block_number: int,
               queue: Queue,
               namespace: Namespace,
               command_event: Event,
               answer_event: Event,
               next_block: Event,
               done_event: Event,
               serial_nr: str = None,
               i2c_speed: float = 100E3,
               spi_turbo: bool = False) -> None:
    """Checks the arguments validity, initializes the device and sets the locks.

    Args:
      mode (:obj:`str`): The communication mode, can be :
        ::

          'SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'

        GPIOs can be driven in any mode, but faster speeds are achievable in
        `GPIO_only` mode.

      block_number (:obj:`int`): The blocks number that was assigned to this
        instance of the class at the first contact with the USB server.

      queue (:obj:`multiprocessing.Queue`): The queue in which the class
        will put its block number so that the USB server knows it is requesting
        control.

      namespace (:obj:`multiprocessing.managers.Namespace`): The Namespace
        object used by the USB server for receiving commands ans sending
        answers.

      command_event (:obj:`multiprocessing.Event`): An event object used by the
        USB server to know when a new command was written by a block.

      answer_event (:obj:`multiprocessing.Event`): An event object used by this
        class to know when the USB server sent back an answer.

      next_block (:obj:`multiprocessing.Event`): An event object, set by the USB
        server to tell the blocks waiting for the control that now is maybe
        their chance.

      done_event (:obj:`multiprocessing.Event`): An event object set by the
        block currently in control of the server to tell it that it is done
        sending commands.

      serial_nr (:obj:`str`, optional): The serial number of the FT232H to
        drive. In `Write_serial_nr` mode, the serial number to be written.

      i2c_speed (:obj:`str`, optional): In I2C mode, the I2C bus clock frequency
        in Hz. Available values are :
        ::

          100E3, 400E3, 1E6.

      spi_turbo (:obj:`str`, optional): Increases the achievable bus speed, but
        may not work with some devices.

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
      raise ValueError("i2c_speed should be in {}".format(list(
        ft232h_i2c_speed.values())))

    self._gpio_low = 0
    self._gpio_high = 0
    self._gpio_dir = 0
    self._retry_count = 8

    self._usb_write_timeout = 5000
    self._usb_read_timeout = 5000

    self._turbo = spi_turbo
    self._i2c_speed = i2c_speed

    self._block_number = block_number
    self._queue = queue
    self._namespace = namespace
    self._command_event = command_event
    self._answer_event = answer_event
    self._next_block = next_block
    self._done_event = done_event

    self._initialize()

    if mode == 'Write_serial_nr':
      self._set_serial_number(serial_nr)
      self.close()

  def _send_server(self, command: Union[list, str]) -> Any:
    """Method for sending commands and receiving answers from the server
    managing the FT232H devices.

    Args:
      command (:obj:`str` or :obj:`list`): The command to send to the server.

    Returns:
      The answer from the server.
    """

    # The timestamp of the last interaction with the server is constantly being
    # saved because the multiprocessing objects timeouts are buggy
    t = time()
    while True:
      try:
        # Communication with the server is allowed only if the block is the one
        # currently in control or if a next_block event is set
        if self._namespace.current_block == self._block_number or \
                self._next_block.wait(timeout=5):
          if self._done_event.is_set():
            # The previous block is done controlling the server but the server
            # hasn't chosen the next block yet
            continue
          # Even if the next_block event is set, only the chosen block is
          # is allowed to communicate
          if self._namespace.current_block == self._block_number:
            # The other blocks will have to wait
            self._next_block.clear()
            # Sending the command
            setattr(self._namespace, 'command' + str(self._block_number),
                    command)
            t = time()
            # Telling that a command was sent
            self._command_event.set()
          else:
            continue
        else:
          # Sometimes the wait method doesn't wait for the given timeout...
          if self._next_block.wait(timeout=5):
            continue
          # Sometimes the timeout check fails twice
          elif time() - t < 2:
            continue
          raise TimeoutError("The server took too long to choose block",
                             (self._block_number,
                              self._namespace.current_block))

        # Retrieving the answer only if the answer_event is set
        if self._answer_event.wait(timeout=5):
          ret = getattr(self._namespace, 'answer' + str(self._block_number))
          t = time()
          # After a CTRL+C or SIGINT event, some of the namespace attributes
          # may be buggy so switching to an "emergency" attribute for receiving
          # the answers
          if ret is None:
            ret = getattr(self._namespace,
                          'answer' + str(self._block_number) + "'")
            if ret is None:
              continue
          self._answer_event.clear()
          # The answer may be an error that should be raised by a block rather
          # than by the server
          if isinstance(ret, Exception):
            raise ret
          # 'ok' is a special answer only received when the block wants to
          # release control
          elif ret == 'ok':
            self._done_event.set()
        else:
          # Again the timeouts are sometimes buggy
          if self._answer_event.wait(timeout=5):
            continue
          elif time() - t < 2:
            continue
          raise TimeoutError("The server took too long to reply",
                             self._block_number, self._namespace.current_block)
        return ret
      except KeyboardInterrupt:
        # In case of a CTRL+C or SIGINT event, the block in control simply
        # resets every event and sends again the command
        if self._namespace.current_block == self._block_number:
          self._command_event.clear()
          self._answer_event.clear()
        continue
      # If the server is down, exiting
      except (BrokenPipeError, ConnectionResetError):
        break

  def _initialize(self) -> None:
    """Initializing the FT232H according to the chosen mode.

    The main differences are for the choice of the clock frequency and
    parameters.
    """

    self._queue.put_nowait(self._block_number)

    # FT232H properties
    fifo_sizes = (1024, 1024)
    latency = 16

    # I2C properties
    if self._ft232h_mode == 'I2C':
      timings = ft232h_i2c_speed[self._i2c_speed]
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

      self._bits_per_word = 8
      self._cshigh = False
      self._no_cs = False
      self._loop = False
      self._lsbfirst = False
      self._max_speed_hz = 400E3
      self._mode = 0
      self._threewire = False
      self._spi_param_changed = True

      self._cs_bit = ft232h_pins['CS']
      self._spi_dir = self._cs_bit | ft232h_pins['SCK'] | ft232h_pins['DO']
      self._spi_mask = self._cs_bit | ft232h_pins['SCK'] | \
          ft232h_pins['DO'] | ft232h_pins['DI']

    else:
      frequency = 400E3

    # Configuring the USB device, interface and endpoints
    try:
      if self._send_server(['is_kernel_driver_active', 0]):
        self._send_server(['detach_kernel_driver', 0])
      self._send_server(['set_configuration'])
    except USBError:
      print("You may have to install the udev-rules for this USB device, "
            "this can be done using the udev_rule_setter utility in the util "
            "folder")
      raise

    self._index, self._in_ep, self._out_ep, self._max_packet_size = \
        self._send_server(['get_active_configuration'])

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
    self._set_bitmode(0, ft232h_server.BitMode.RESET)

    # Set latency timer
    self._set_latency_timer(latency)

    # Set chunk size and invalidate all remaining data
    self._writebuffer_chunksize = fifo_sizes[0]
    self._readoffset = 0
    self._readbuffer = bytearray()
    self._readbuffer_chunksize = min(fifo_sizes[0], fifo_sizes[1],
                                     self._max_packet_size)

    # Reset feature mode
    self._set_bitmode(0, ft232h_server.BitMode.RESET)
    # Drain buffers
    self._purge_buffers()
    # Disable event and error characters
    if self._ctrl_transfer_out(ft232h_sio_req['set_event_char'], 0):
      raise IOError('Unable to set event char')
    if self._ctrl_transfer_out(ft232h_sio_req['set_error_char'], 0):
      raise IOError('Unable to set error char')

    # Enable MPSSE mode
    if self._ft232h_mode == 'GPIO_only':
      self._set_bitmode(0xFF, ft232h_server.BitMode.BITBANG)
    else:
      self._set_bitmode(self._direction, ft232h_server.BitMode.MPSSE)

    # Configure clock
    if self._ft232h_mode == 'I2C':
      # Note that bus frequency may differ from clock frequency, when
      # 3-phase clock is enabled
      self._set_frequency(3 * frequency / 2)
    else:
      self._set_frequency(frequency)

    # Configure pins
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
    self._write_data(bytearray((ft232h_cmds['loopback_end'],)))
    # Validate MPSSE
    bytes_ = bytes(self._read_data_bytes(2))
    if (len(bytes_) >= 2) and (bytes_[0] == '\xfa'):
      raise IOError("Invalid command @ %d" % bytes_[1])

    # I2C-specific settings
    if self._ft232h_mode == 'I2C':
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

    self._send_server('stop')

  @staticmethod
  def _compute_delay_cycles(value: float) -> int:
    """Approximates the number of clock cycles over a given delay.

    Args:
      value (:obj:`float`): delay (in seconds)

    Returns:
      Number of clock cycles
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
      latency (:obj:`int`): latency (in milliseconds)
    """

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
      frequency (:obj:`float`): Desired bus frequency (in Hz)

    Returns:
      Actual bus frequency
    """

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
      bitmask (:obj:`int`): Mask for choosing the driven GPIOs.
      mode (:class:`BitMode`): Bitbang mode to be set.
    """

    mask = sum(ft232h_server.BitMode)
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
      reqtype (:obj:`int`): bmRequest
      value (:obj:`int`): wValue
      data (:obj:`bytes`): payload

    Returns:
      Number of bytes actually written
    """

    try:
      return self._send_server(['ctrl_transfer', Ftdi_req_out, reqtype, value,
                                self._index, bytearray(data),
                                self._usb_write_timeout])
    except USBError as ex:
      raise IOError('UsbError: %s' % str(ex))

  def _set_serial_number(self, serial_number: str) -> None:
    """(Over)Writes the serial number.

    Writes the desired serial number to the EEPROM. It is then accessible to
    USB commands as a string descriptor. Also sets the manufacturer and
    product string descriptors.

    Args:
      serial_number (:obj:`str`): Serial number to be written in the EEPROM
    """

    self._queue.put_nowait(self._block_number)

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
        buf = self._send_server(['ctrl_transfer', Ftdi_req_in,
                                 ft232h_sio_req['read_eeprom'], 0, word_addr, 2,
                                 self._usb_read_timeout])
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
      out = self._send_server(['ctrl_transfer', Ftdi_req_out,
                               ft232h_sio_req['write_eeprom'], word, addr >> 1,
                               b'', self._usb_write_timeout])
      if out:
        raise IOError('EEPROM Write Error @ %d' % addr)
      addr += 2

    self._send_server('stop')

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
          length = self._send_server(['write', self._in_ep,
                                      data[offset:offset + write_size],
                                      self._usb_write_timeout])
        except USBError:
          raise

        if length <= 0:
          raise USBError("Usb bulk write error")
        offset += length
      return offset
    except USBError:
      print("An error occurred while writing to USB")
      raise

  def _read_data_bytes(self,
                       size: int,
                       attempt: int = 2,
                       request_gen: Callable = None) -> bytes:
    """Reads data from the FT232H.

    Reads data from the FTDI interface. The data buffer is rebuilt from
    chunk-sized blocks received over the USB bus. The FTDI device always sends
    internal status bytes, which are stripped out as not part of the data
    payload.

    Args:
      size (:obj:`int`): The number of bytes to receive from the device
      attempt (:obj:`int`): Attempt cycle count
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
            tempbuf = self._send_server(['read', self._out_ep,
                                         self._readbuffer_chunksize,
                                         self._usb_read_timeout])
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
                req_size -= length - 2
                if req_size > 0:
                  cmd = request_gen(req_size)
                  if cmd:
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
      print("An error occurred while writing to USB")
      raise
    # never reached
    raise ValueError("Internal error")

  @property
  def _clk_hi_data_lo(self) -> tuple:
    """Returns the MPSSE command for driving CLK line high and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SCL'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_input(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and listening to SDA
       line, while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            ft232h_pins['SCL'] | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_hi(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            ft232h_pins['SDAO'] | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _clk_lo_data_lo(self) -> tuple:
    """Returns the MPSSE command for driving CLK line low and SDA line low,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _idle(self) -> tuple:
    """Returns the MPSSE command for driving CLK line high and SDA line high,
       while preserving the GPIO outputs."""

    return (ft232h_cmds['set_bits_low'],
            self._i2c_dir | self._gpio_low,
            self._i2c_dir | (self._gpio_dir & 0xFF))

  @property
  def _start(self) -> tuple:
    """Returns the MPSSE command for issuing and I2C start condition."""

    return self._clk_hi_data_lo * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta

  @property
  def _stop(self) -> tuple:
    """Returns the MPSSE command for issuing and I2C stop condition."""

    return self._clk_lo_data_hi * self._ck_hd_sta + \
        self._clk_lo_data_lo * self._ck_hd_sta + \
        self._clk_hi_data_lo * self._ck_su_sto + \
        self._idle * self._ck_idle

  def _do_prolog(self, i2caddress: int) -> None:
    """Sends the MPSSE commands for starting an I2C transaction.

    Args:
      i2caddress (:obj:`int`): I2C address of the slave
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
      out (:obj:`list`): List of bytes to write
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
    """
    Sends the MPSSE commands for reading bytes from an I2C slave, and then
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
    chunk_size = min(tx_count, chunk_size)
    chunks = []
    rem = readlen
    if rem > chunk_size:
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
        buf = self._read_data_bytes(rem, 80, _write_command_gen)
        chunks.append(buf)
        rem -= len(buf)
    else:
      cmd = bytearray()
      cmd.extend(read_not_last * (rem - 1))
      cmd.extend(read_last)
      cmd.extend((ft232h_cmds['send_immediate'],))
      size = rem
      self._write_data(cmd)
      buf = self._read_data_bytes(size, 8)
      chunks.append(buf)
    return bytearray(b''.join(chunks))

  def _send_check_ack(self, cmd: bytearray) -> None:
    """Actually sends the MPSSE commands generated by :meth:`_do_prolog` and
    :meth:`_do_write` methods, and checks whether the slave ACKs it.

    Args:
      cmd (:obj:`bytearray`): The MPSSE commands to send
    """

    # SCL low, SDA high-Z
    cmd.extend(self._clk_lo_data_hi)
    # read SDA (ack from slave)
    cmd.extend((ft232h_cmds['read_bits_PVE_MSB'], 0))
    cmd.extend((ft232h_cmds['send_immediate'],))
    self._write_data(cmd)
    ack = self._read_data_bytes(1, 8)
    if not ack:
      raise IOError('No answer from FTDI')
    if ack[0] & 0x01:
      raise IOError('NACK from slave')

  def _write_i2c(self, address: int, out: list) -> None:
    """Writes bytes to an I2C slave.

    Args:
      address (:obj:`int`): I2C address of the slave
      out (:obj:`list`): List of bytes to send
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
        self._write_data(bytearray(self._stop))

  def _exchange_i2c(self,
                    address: int,
                    out: list,
                    readlen: int = 0) -> bytearray:
    """Writes bytes to an I2C slave, and then reads a given number of bytes
    from this same slave.

    Args:
      address (:obj:`int`): I2C address of the slave
      out (:obj:`list`): List of bytes to send
      readlen (:obj:`int`): Number of bytes to read

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
        data = self._do_read(readlen)
        return data
      except IOError:
        retries -= 1
        if not retries:
          raise
      finally:
        self._write_data(bytearray(self._stop))

  def write_byte(self, i2c_addr: int, value: int) -> None:
    """Writes a single byte to an I2C slave, in register 0.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=0x00,
                              data=[value & 0xFF])

  def write_byte_data(self, i2c_addr: int, register: int, value: int) -> None:
    """Writes a single byte to an I2C slave, in the specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the register to be written
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[value & 0xFF])

  def write_word_data(self, i2c_addr: int, register: int, value: int) -> None:
    """Writes 2 bytes to an I2C slave from a single int value, starting at the
    specified register.

    Depending on the size of the registers, the next register may be written as
    well.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      value (:obj:`int`): The value to write
    """

    self.write_i2c_block_data(i2c_addr=i2c_addr,
                              register=register,
                              data=[(value >> 8) & 0xFF, value & 0xFF])

  def write_block_data(self, i2c_addr: int, register: int, data: list) -> None:
    """Actually calls :meth:`write_i2c_block_data`.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      data (:obj:`list`): List of bytes to write
    """

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
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be written
      data (:obj:`list`): List of bytes to write
    """

    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")

    self._write_i2c(address=i2c_addr,
                    out=[register] + data)

    self._send_server('stop')

  def read_byte(self, i2c_addr: int) -> int:
    """Reads a single byte from an I2C slave, from the register `0`.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave

    Returns:
      Value of the read register
    """

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=0x00,
                                      length=1)[0]
    except IndexError:
      print("No data to read")
      raise

  def read_byte_data(self, i2c_addr: int, register: int) -> int:
    """Reads a single byte from an I2C slave, from the specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the register to be read

    Returns:
      Value of the read register
    """

    try:
      return self.read_i2c_block_data(i2c_addr=i2c_addr,
                                      register=register,
                                      length=1)[0]
    except IndexError:
      print("No data to read")
      raise

  def read_word_data(self, i2c_addr: int, register: int) -> int:
    """Reads 2 bytes from an I2C slave, starting at the specified register, and
    returns them as one single value.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be read

    Returns:
      Value of the read registers
    """

    try:
      ret = self.read_i2c_block_data(i2c_addr=i2c_addr,
                                     register=register,
                                     length=2)
      return (ret[0] << 8) & ret[1]
    except IndexError:
      print("Not enough data to read")
      raise

  def read_i2c_block_data(self,
                          i2c_addr: int,
                          register: int,
                          length: int) -> list:
    """Reads a given number of bytes from an I2C slave, starting at the
    specified register.

    Args:
      i2c_addr (:obj:`int`): I2C address of the slave
      register (:obj:`int`): Index of the first register to be read
      length (:obj:`int`): Number of bytes to read

    Returns:
      Values of read registers as a :obj:`list`
    """

    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'I2C':
      raise ValueError("Method only available in I2C mode")
    if not 0 <= i2c_addr <= 127:
      raise ValueError("Incorrect i2c address, should be between 0 and 127")
    if not length >= 0:
      raise ValueError("length should be a positive integer")

    if length == 0:
      self._send_server('stop')
      return []
    ret = [byte for byte in self._exchange_i2c(address=i2c_addr,
                                               out=[register],
                                               readlen=length)]
    self._send_server('stop')
    return ret

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
    self._cshigh = value

  @property
  def loop(self) -> bool:
    """If :obj:`True`, the loopback mode is enabled."""

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    return self._loop

  @loop.setter
  def loop(self, value: bool) -> None:
    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'SPI':
      raise ValueError("Attribute only available in SPI mode")
    if not isinstance(value, bool):
      raise TypeError("loop should be either True or False")
    if value:
      self._write_data(bytearray((ft232h_cmds['loopback_start'],)))
    else:
      self._write_data(bytearray((ft232h_cmds['loopback_end'],)))
    self._loop = value

    self._send_server('stop')

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
    self._queue.put_nowait(self._block_number)

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
    self._max_speed_hz = value

    self._send_server('stop')

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
    self._threewire = value

  def _exchange_spi(self, readlen: int, out: list, start: bool,
                    stop: bool, duplex: bool) -> bytes:
    """Exchanges bytes with an SPI slave.

    Can read and/or write data, in a sequential or simultaneous way. Also
    manages the CS line.

    Args:
      readlen (:obj:`int`): Number of bytes to read. If 0, no reading is
        performed.
      out (:obj:`list`): List of bytes to write. If empty, no writing is
        performed.
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        exchanging data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        exchanging data, and remains in its previous state.
      duplex (:obj:`int`): If :obj:`True`, the data is read and written
        simultaneously. If :obj:`False`, writes then reads in a sequential way.

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

  def readbytes(self, len: int, start: bool = True, stop: bool = True) -> list:
    """Reads the specified number of bytes from an SPI slave.

    Args:
      len (:obj:`int`): Number of bytes to read
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.

    Returns:
      List of read bytes
    """

    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    ret = [byte for byte in self._exchange_spi(readlen=len,
                                               out=[],
                                               start=start,
                                               stop=stop,
                                               duplex=False)]
    self._send_server('stop')
    return ret

  def writebytes(self,
                 values: list,
                 start: bool = True,
                 stop: bool = True) -> None:
    """Write bytes from a list to an SPI slave.

    Args:
      values (:obj:`list`): List of bytes to write
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.
    """

    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    self._exchange_spi(readlen=0,
                       out=values,
                       start=start,
                       stop=stop,
                       duplex=False)
    self._send_server('stop')

  def writebytes2(self,
                  values: list,
                  start: bool = True,
                  stop: bool = True) -> None:
    """Actually calls the :meth:`writebytes` method with the same arguments."""

    self.writebytes(values=values,
                    start=start,
                    stop=stop)

  def xfer(self,
           values: list,
           speed: float = None,
           delay: float = 0.0,
           bits: int = 8,
           start: bool = True,
           stop: bool = True) -> list:
    """Simultaneously reads and write bytes to an SPI slave.

    The number of bytes to read is equal to the number of bytes in the write
    buffer.

    Args:
      values (:obj:`list`): List of bytes to write
      speed (:obj:`float`): Sets the bus clock frequency before issuing the
        command (in Hz)
      delay (:obj:`float`): Not implemented, should be 0.0
      bits (:obj:`int`):  Not implemented, should be 8
      start (:obj:`bool`): If :obj:`False`, the CS line is not driven before
        reading data, and remains in its previous state.
      stop (:obj:`bool`): If :obj:`False`, the CS line is not driven after
        reading data, and remains in its previous state.

    Returns:
      List of read bytes
    """

    self._queue.put_nowait(self._block_number)

    if self._ft232h_mode != 'SPI':
      raise ValueError("Method only available in SPI mode")
    if bits != 8:
      raise ValueError("Only bits=8 is implemented")
    if delay != 0.0:
      raise ValueError("The delay parameter is not currently implemented")

    if speed != self.max_speed_hz and speed is not None:
      self.max_speed_hz = speed

    ret = [byte for byte in self._exchange_spi(readlen=len(values),
                                               out=values,
                                               start=start,
                                               stop=stop,
                                               duplex=True)]
    self._send_server('stop')
    return ret

  def xfer2(self,
            values: list,
            speed: float = 6.0E6,
            delay: float = 0.0,
            bits: int = 8,
            start: bool = True,
            stop: bool = True) -> list:
    """Actually calls the :meth:`xfer` method with the same arguments."""

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
            stop: bool = True) -> list:
    """Actually calls the :meth:`xfer` method with the same arguments."""

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

    A :obj:`True` bit represents an output pin, a :obj:`False` bit an input pin.

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
    """Sends the MPSSE commands for reading all the FT232H pins, and returns the
    bitmap of read values. Values are determined using 3.3V logic.

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

  def get_gpio(self, gpio_str: str) -> int:
    """Reads the 3.3V-logic voltage value of the specified pin.

    Args:
      gpio_str (:obj:`str`): Name of the GPIO to be read

    Returns:
      3.3V-logic value corresponding to the input voltage
    """

    self._queue.put_nowait(self._block_number)

    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    gpio_bit = ft232h_pin_nr[gpio_str]
    if not self._gpio_all_pins & gpio_bit:
      raise ValueError("Cannot use pin {} as a GPIO".format(gpio_str))

    # Changing the _direction and _gpio_dir bitfields
    if self._direction & gpio_bit:
      self._gpio_dir &= 0xFFFF - gpio_bit

    ret = self._read_gpio_raw() & gpio_bit
    self._send_server('stop')
    return ret

  def set_gpio(self, gpio_str: str, value: int) -> None:
    """Sets the specified GPIO as an output and sets its output value.

    Args:
      gpio_str (:obj:`str`): Name of the GPIO to be set
      value (:obj:`int`): 1 for setting the GPIO high, 0 for setting it low
    """

    self._queue.put_nowait(self._block_number)

    if value not in [0, 1]:
      raise ValueError("value should be either 0 or 1")
    if gpio_str not in ft232h_pin_nr:
      raise ValueError("gpio_id should be in {}".format(
        list(ft232h_pin_nr.values())))
    gpio_bit = ft232h_pin_nr[gpio_str]
    if not self._gpio_all_pins & gpio_bit:
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

    self._send_server('stop')

  def close(self) -> None:
    """Closes the FTDI interface/port."""

    self._queue.put_nowait(self._block_number)

    if self._send_server('close?'):

      if self._send_server('_ctx.handle'):
        try:
          self._set_bitmode(0, ft232h_server.BitMode.RESET)
          self._send_server([None, 'release_interface', 'dev',
                             self._index - 1])
        except (IOError, ValueError, USBError):
          pass
        try:
          self._send_server(['attach_kernel_driver', self._index - 1])
        except (NotImplementedError, USBError):
          pass
      self._send_server([None, 'dispose_resources', 'dev'])

    self._send_server('farewell')
