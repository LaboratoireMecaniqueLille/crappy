# coding: utf-8

from collections import namedtuple
from struct import unpack
from typing import Union, Optional, Literal
from collections.abc import Callable
from _io import FileIO
from multiprocessing.synchronize import RLock
from multiprocessing.sharedctypes import Synchronized
from time import time, sleep
import logging
from contextlib import contextmanager
import signal

from .ft232h import FT232H
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

ft232h_eeprom_size = 256
ft232h_tx_empty_bits = 0x60

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

ft232h_i2c_speed = {100E3: ft232h_i2c_timings(4.0E-6, 4.7E-6, 4.0E-6, 4.7E-6),
                    400E3: ft232h_i2c_timings(0.6E-6, 0.6E-6, 0.6E-6, 1.3E-6),
                    1E6: ft232h_i2c_timings(0.26E-6, 0.26E-6, 0.26E-6, 0.5E-6)}


class DelayedKeyboardInterrupt:
  """This class implements a context manager for temporarily disabling the
  :exc:`KeyboardInterrupt` and storing any exception received in the meantime.

  It is meant to avoid having a I2C or SPI communication interrupted, which 
  could cause devices to bug and not be able to properly finish.
  
  .. versionadded:: 2.0.0
  """

  def __enter__(self) -> None:
    """Enters the context and sets :meth:`_handler` as the new handler for
    SIGINT signals."""

    self._signal_received = None
    self._prev_handler = signal.signal(signal.SIGINT, self._handler)

  def __exit__(self, _, __, ___) -> None:
    """Exits the context, sets the previous handler back, and handles any
    SIGINT signal received while in the context."""

    signal.signal(signal.SIGINT, self._prev_handler)
    if self._signal_received is not None:
      self._prev_handler(*self._signal_received)

  def _handler(self, sig, frame) -> None:
    """Handler that just stores the received SIGINT while in the context."""

    self._signal_received = (sig, frame)


class FT232HServer(FT232H):
  """A class for controlling FTDI's USB to Serial FT232H.

  This class is very similar to the :class:`~crappy.tool.ft232h.FT232H` except 
  it doesn't directly instantiate the USB device nor send commands to it 
  directly. Instead, the commands are sent to a 
  :class:`~crappy.tool.ft232h.USBServer` managing communication with the FT232H 
  device(s).

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
  .. versionchanged:: 2.0.0 renamed from *ft232h_server* to *FT232HServer*
  """

  def __init__(self,
               mode: Literal['SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'],
               block_index: int,
               current_block: Synchronized,
               command_file: FileIO,
               answer_file: FileIO,
               block_lock: RLock,
               shared_lock: RLock,
               serial_nr: Optional[str] = None,
               i2c_speed: float = 100E3,
               spi_turbo: bool = False) -> None:
    """Checks the argument validity and initializes the device.

    Args:
      mode: The communication mode as a :obj:`str`, can be :
        ::

          'SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'

        GPIOs can be driven in any mode, but faster speeds are achievable in
        `GPIO_only` mode.
      block_index: The index the :class:`~crappy.blocks.Block` driving this 
        FT232HServer instance has been assigned by the 
        :class:`~crappy.tool.ft232h.USBServer`, as an :obj:`int`.

        .. versionchanged:: 2.0.0 renamed from *block_number* to *block_index*
      current_block: The handle to a shared :obj:`multiprocessing.Value` 
        indicating which :class:`~crappy.blocks.Block` can currently
        communicate with the :class:`~crappy.tool.ft232h.USBServer`.

        .. versionchanged:: 2.0.0
           renamed from *current_file* to *current_block*
      command_file: A file in which the current command to be executed by the
        USB server is written.
      answer_file: A file in which the answer to the current command is
        written.
      block_lock: A :obj:`multiprocessing.Lock` assigned to this 
        :class:`~crappy.blocks.Block` only, for signaling the 
        :class:`~crappy.tool.ft232h.USBServer` when the command has been 
        written in the command_file.
      shared_lock: A :obj:`multiprocessing.Lock` common to all the 
        :class:`~crappy.blocks.Block` that allows the one Block holding it to 
        communicate with the :class:`~crappy.tool.ft232h.USBServer`.

        .. versionchanged:: 2.0.0 renamed from *current_lock* to *shared_lock*
      serial_nr: The serial number of the FT232H to drive, as a :obj:`str`. In
        `Write_serial_nr` mode, the serial number to be written.
      i2c_speed: In I2C mode, the I2C bus clock frequency in Hz. Available
        values are :
        ::

          100E3, 400E3, 1E6

        or any value between `10kHz` and `100kHz`. Lowering below the default
        value may solve I2C clock stretching issues on some devices.
      spi_turbo: Increases the achievable bus speed in SPI mode, but may not
        work with some devices.

    Note:
      - **CS pin**:
        The CS pin for selecting SPI devices is always `D3`. This pin is
        reserved and cannot be used as a GPIO. If you want to drive the CS line
        manually, it is possible not to drive the CS pin by setting the SPI
        parameter :attr:`~crappy.tool.ft232h.ft232h_server.FT232HServer.no_cs`
        to :obj:`True` and to drive the CS line from a GPIO instead.

      - ``mode``:
        It is not possible to simultaneously control slaves over SPI and I2C,
        due to different hardware requirements for the two protocols. Trying to
        do so will most likely raise an error or lead to inconsistent behavior.
    """

    self._block_index = block_index
    self._current_block = current_block
    self._command_file = command_file
    self._answer_file = answer_file
    self._block_lock = block_lock
    self._shared_lock = shared_lock

    super().__init__(mode=mode, serial_nr=serial_nr, i2c_speed=i2c_speed,
                     spi_turbo=spi_turbo)

    self._nb_attempt_1 = 80
    self._nb_attempt_2 = 2

  def _handle_command(self, command: list) -> bytes:
    """Parses the command and sends it to the device.

    Args:
      command: The :obj:`list` containing the command type and the arguments if
        any.

    Returns:
      The index of the command.
    """

    # Control transfer out
    if command[0] == 'ctrl_transfer_out':
      value = b','.join((b'00', str(command[1]).encode(),
                         str(command[2]).encode(), str(command[3]).encode(),
                         str(command[4]).encode(), bytes(command[5]),
                         str(command[6]).encode()))
      cmd = b'00'
    # Control transfer in
    elif command[0] == 'ctrl_transfer_in':
      value = b','.join((b'01', str(command[1]).encode(),
                         str(command[2]).encode(), str(command[3]).encode(),
                         str(command[4]).encode(), str(command[5]).encode(),
                         str(command[6]).encode()))
      cmd = b'01'
    # Write operation
    elif command[0] == 'write':
      value = b','.join((b'02', str(command[1]).encode(), bytes(command[2]),
                         str(command[3]).encode()))
      cmd = b'02'
    # Read operation
    elif command[0] == 'read':
      value = b','.join((b'03', str(command[1]).encode(),
                         str(command[2]).encode(), str(command[3]).encode()))
      cmd = b'03'
    # Checks whether the kernel driver is active
    # It doesn't actually interact with the device
    elif command[0] == 'is_kernel_driver_active':
      value = b','.join((b'04', str(command[1]).encode()))
      cmd = b'04'
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == 'detach_kernel_driver':
      value = b','.join((b'05', str(command[1]).encode()))
      cmd = b'05'
    # Sets the device configuration
    elif command[0] == 'set_configuration':
      value = b'06'
      cmd = b'06'
    # Custom command getting information from the current configuration
    elif command[0] == 'get_active_configuration':
      value = b'07'
      cmd = b'07'
    # Should the block close the device when leaving ?
    # It doesn't actually interact with the device
    elif command[0] == 'close?':
      value = b'08'
      cmd = b'08'
    # Checks whether the internal resources have been released or not
    # It doesn't actually interact with the device
    elif command[0] == '_ctx.handle':
      value = b'09'
      cmd = b'09'
    # Releases the USB interface
    # It doesn't actually interact with the device
    elif command[0] == 'release_interface':
      value = b','.join((b'10', str(command[1]).encode()))
      cmd = b'10'
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == 'attach_kernel_driver':
      value = b','.join((b'11', str(command[1]).encode()))
      cmd = b'11'
    # Releases all the resources used by :mod:`pyusb` for a given device
    # It doesn't actually interact with the device
    elif command[0] == 'dispose_resources':
      value = b'12'
      cmd = b'12'
    # Registers a block as gone
    # It doesn't actually interact with the device
    elif command[0] == 'farewell':
      value = b'13'
      cmd = b'13'
    else:
      raise ValueError("Wrong command type !")

    self.log(logging.DEBUG, f"Writing command {value} to command buffer")
    self._command_file.write(value)

    return cmd

  def _send_server(self, command: list) -> Union[int, bytes, None,
                                                 tuple[int, ...]]:
    """Sends a command to the server and gets the corresponding answer.

    Args:
      command: A :obj:`list` containing the command type as a first element,
        and then the arguments if any.
    """

    # Disabling KeyboardInterrupt to avoid unexpected behavior upon CTRL+C
    with DelayedKeyboardInterrupt():
      self.log(logging.DEBUG, "KeyBoardInterrupt disabled")

      # Acquiring the shared lock to get control over the server
      with self.acquire_timeout(self._shared_lock, 1) as acquired:
        if acquired:
          self.log(logging.DEBUG, "Acquired shared lock")

          # Acquiring the block lock to indicate the command is being written
          with self.acquire_timeout(self._block_lock, 1) as acq:
            if acq:
              self.log(logging.DEBUG, "Acquired block lock")
              self.log(logging.DEBUG,
                       f"Writing {str(self._block_index).encode()} as the "
                       f"current block index")
              self._current_block.value = self._block_index

              self._command_file.seek(0)
              self._command_file.truncate(0)
              # Writing the command and its arguments to the command file
              cmd = self._handle_command(command)

            else:
              raise TimeoutError("Could not acquire the block lock in 1s")

          # Waiting for the server to write the answer
          # If using only the lock, it could be re-acquired by this class
          # without the server having a chance to get it
          t = time()
          while not self._answer_file.tell():
            sleep(0.00001)
            if time() - t > 1:
              raise TimeoutError("No answer from the USB server after 1s")

          # The lock can be re-acquired once the answer has been written
          with self.acquire_timeout(self._block_lock, 1) as acq:
            if acq:
              self.log(logging.DEBUG, "Acquired the block lock")
              # Reading the answer
              self._answer_file.seek(0)
              answer: list[bytes] = self._answer_file.read().split(b',')
              self.log(logging.DEBUG, f"Read {answer} from the answer file")
              self._answer_file.seek(0)
              self._answer_file.truncate(0)

              if cmd != answer[0]:
                raise IOError("Got an answer for the command of another block")

              # The different answers have to be handled in various ways
              if command[0] in ['ctrl_transfer_out',
                                'write',
                                'is_kernel_driver_active',
                                'close?',
                                '_ctx.handle']:
                return int(answer[1])
              if command[0] in ['ctrl_transfer_in', 'read']:
                return answer[1]
              elif command[0] == 'get_active_configuration':
                return tuple(int(rep) for rep in answer[1:])

              return

            else:
              raise TimeoutError("Could not acquire the block lock in 1s")

        else:
          raise TimeoutError("Could not acquire the shared lock in 1s")

    self.log(logging.DEBUG, "KeyBoardInterrupt re-enabled")

  def _initialize(self) -> None:
    """Initializing the FT232H according to the chosen mode.

    The main differences are for the choice of the clock frequency and
    parameters.
    """

    # FT232H properties
    fifo_sizes = (1024, 1024)
    latency = 2

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
      self.log(logging.INFO, "Setting USB configuration for the FT232H")
      self._send_server(['set_configuration'])
    except USBError:
      self.log(logging.ERROR,
               "Could not set USB device configuration !\nYou may have to "
               "install the udev-rules for this USB device, this can be done "
               "using the udev_rule_setter utility in the util folder")
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
      return self._send_server(['ctrl_transfer_out', Ftdi_req_out, reqtype,
                                value, self._index, bytearray(data),
                                self._usb_write_timeout])
    except USBError as ex:
      raise IOError('UsbError: %s' % str(ex))

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
      size: The number of bytes to receive from the device
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
            tempbuf = self._send_server(['read', self._out_ep,
                                         self._readbuffer_chunksize,
                                         self._usb_read_timeout])
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
        buf = self._send_server(['ctrl_transfer_in', Ftdi_req_in,
                                 ft232h_sio_req['read_eeprom'], 0, word_addr,
                                 2, self._usb_read_timeout])
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
      out = self._send_server(['ctrl_transfer_out', Ftdi_req_out,
                               ft232h_sio_req['write_eeprom'], word, addr >> 1,
                               b'', self._usb_write_timeout])
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
      self.log(logging.ERROR, "An error occurred while writing to USB device")
      raise

  def close(self) -> None:
    """Closes the FTDI interface/port."""

    if self._send_server(['close?']):

      self.log(logging.INFO, "Closing the USB connection to the FT232H")
      if self._send_server(['_ctx.handle']):
        try:
          self._set_bitmode(0, FT232H.BitMode.RESET)
          self._send_server(['release_interface', self._index - 1])
        except (IOError, ValueError, USBError):
          pass
        try:
          self._send_server(['attach_kernel_driver', self._index - 1])
        except (NotImplementedError, USBError):
          pass
      self.log(logging.INFO, "Releasing the USB resources")
      self._send_server(['dispose_resources'])

    self._send_server(['farewell'])

  @staticmethod
  @contextmanager
  def acquire_timeout(lock: RLock, timeout: float) -> bool:
    """Short context manager for acquiring a :obj:`multiprocessing.Lock` with a
    specified timeout.

    Args:
      lock: The lock to acquire.
      timeout: The timeout for acquiring the Lock, as a :obj:`float`.

    Returns:
      :obj:`True` if the Lock was successfully acquired, :obj:`False`
      otherwise.
    """

    ret = False
    try:
      ret = lock.acquire(timeout=timeout)
      yield ret
    finally:
      if ret:
        lock.release()
