# coding: utf-8

from collections import namedtuple
from struct import unpack
from typing import Union, List, Tuple, Optional, Callable
from _io import FileIO
from multiprocessing.synchronize import RLock
from time import time, sleep
from warnings import warn

from .ft232h import ft232h
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


class ft232h_server(ft232h):
  """A class for controlling FTDI's USB to Serial FT232H.

  This class is very similar to the :class:`ft232h` except it doesn't
  directly instantiate the USB device nor send commands to it directly.
  Instead, the commands are sent to a USB server managing communication with
  the different FT232H devices.

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

      $ echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"0403\\", \
MODE=\\"0666\\\"" | sudo tee ftdi.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.

  Important:
    For controlling several FT232H from the same computer, it is first
    necessary to set their USB serial numbers. Otherwise, an error will be
    raised. This can be done using the crappy utility
    ``Set_ft232h_serial_nr.py``.
  """

  def __init__(self,
               mode: str,
               block_number: int,
               current_file: FileIO,
               command_file: FileIO,
               answer_file: FileIO,
               block_lock: RLock,
               current_lock: RLock,
               serial_nr: Optional[str] = None,
               i2c_speed: float = 100E3,
               spi_turbo: bool = False) -> None:
    """Checks the argument validity and initializes the device.

    Args:
      mode: The communication mode, can be :
        ::

          'SPI', 'I2C', 'GPIO_only', 'Write_serial_nr'

        GPIOs can be driven in any mode, but faster speeds are achievable in
        `GPIO_only` mode.
      block_number: The index the block driving this ft232h_server instance has
        been assigned.
      current_file: A file in which the index of the block currently allowed to
        drive the USB server is written.
      command_file: A file in which the current command to be executed by the
        USB server is written.
      answer_file: A file in which the answer to the current command is
        written.
      block_lock: A lock assigned to this block only, for signaling the USB
        server when the command has been written in the command_file.
      current_lock: A lock common to all the blocks that allows the one block
        holding it to communicate with the USB server.
      serial_nr (:obj:`str`, optional): The serial number of the FT232H to
        drive. In `Write_serial_nr` mode, the serial number to be written.
      i2c_speed: In I2C mode, the I2C bus clock frequency in Hz. Available
        values are :
        ::

          100E3, 400E3, 1E6

        or any value between `10kHz` and `100kHz`. Lowering below the default
        value may solve I2C clock stretching issues on some devices.

      spi_turbo: Increases the achievable bus speed, but may not work with some
        devices.

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

    warn("The ft232h_server class will be renamed to FT232HServer in version "
         "2.0.0", DeprecationWarning)
    warn("The block_number argument will be renamed to block_index in version "
         "2.0.0", DeprecationWarning)
    warn("The current_lock argument will be renamed to shared_lock in version "
         "2.0.0", DeprecationWarning)

    self._block_number = block_number
    self._current_file = current_file
    self._command_file = command_file
    self._answer_file = answer_file
    self._block_lock = block_lock
    self._current_lock = current_lock

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
      self._command_file.write(b'00' + b',' +
                               str(command[1]).encode() + b',' +
                               str(command[2]).encode() + b',' +
                               str(command[3]).encode() + b',' +
                               str(command[4]).encode() + b',' +
                               bytes(command[5]) + b',' +
                               str(command[6]).encode())
      cmd = b'00'
    # Control transfer in
    elif command[0] == 'ctrl_transfer_in':
      self._command_file.write(b'01' + b',' +
                               str(command[1]).encode() + b',' +
                               str(command[2]).encode() + b',' +
                               str(command[3]).encode() + b',' +
                               str(command[4]).encode() + b',' +
                               str(command[5]).encode() + b',' +
                               str(command[6]).encode())
      cmd = b'01'
    # Write operation
    elif command[0] == 'write':
      self._command_file.write(b'02' + b',' +
                               str(command[1]).encode() + b',' +
                               bytes(command[2]) + b',' +
                               str(command[3]).encode())
      cmd = b'02'
    # Read operation
    elif command[0] == 'read':
      self._command_file.write(b'03' + b',' +
                               str(command[1]).encode() + b',' +
                               str(command[2]).encode() + b',' +
                               str(command[3]).encode())
      cmd = b'03'
    # Checks whether the kernel driver is active
    # It doesn't actually interact with the device
    elif command[0] == 'is_kernel_driver_active':
      self._command_file.write(b'04' + b',' +
                               str(command[1]).encode())
      cmd = b'04'
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == 'detach_kernel_driver':
      self._command_file.write(b'05' + b',' +
                               str(command[1]).encode())
      cmd = b'05'
    # Sets the device configuration
    elif command[0] == 'set_configuration':
      self._command_file.write(b'06')
      cmd = b'06'
    # Custom command getting information from the current configuration
    elif command[0] == 'get_active_configuration':
      self._command_file.write(b'07')
      cmd = b'07'
    # Should the block close the device when leaving ?
    # It doesn't actually interact with the device
    elif command[0] == 'close?':
      self._command_file.write(b'08')
      cmd = b'08'
    # Checks whether the internal resources have been released or not
    # It doesn't actually interact with the device
    elif command[0] == '_ctx.handle':
      self._command_file.write(b'09')
      cmd = b'09'
    # Releases the USB interface
    # It doesn't actually interact with the device
    elif command[0] == 'release_interface':
      self._command_file.write(b'10' + b',' +
                               str(command[1]).encode())
      cmd = b'10'
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == 'attach_kernel_driver':
      self._command_file.write(b'11' + b',' +
                               str(command[1]).encode())
      cmd = b'11'
    # Releases all the resources used by :mod:`pyusb` for a given device
    # It doesn't actually interact with the device
    elif command[0] == 'dispose_resources':
      self._command_file.write(b'12')
      cmd = b'12'
    # Registers a block as gone
    # It doesn't actually interact with the device
    elif command[0] == 'farewell':
      self._command_file.write(b'13')
      cmd = b'13'
    else:
      raise ValueError("Wrong command type !")

    return cmd

  def _send_server(self, command: list) -> Union[int, bytes, None,
                                                 Tuple[int, ...]]:
    """Sends a command to the server and gets the corresponding answer.

    Args:
      command: A :obj:`list` containing the command type as a first element,
        and then the arguments if any.
    """

    raise_kbi = False  # Flag for postponing the raise of the exception
    retries = 3  # Number of retries for acquiring the lock
    while True:
      try:
        # Acquires the lock assigned to this block only
        if not self._block_lock.acquire(timeout=1):
          retries -= 1
          if not retries:
            raise TimeoutError("Couldn't acquire the lock in a reasonable "
                               "delay !")
          continue
        # Acquires the lock common to all blocks, to start communicating with
        # the server
        if self._current_lock.acquire(timeout=1):
          while True:
            release = True  # Should the common lock be released ?
            try:

              # Writing the block number in the file
              self._current_file.seek(0)
              self._current_file.truncate(0)
              self._current_file.write(str(self._block_number).encode())

              self._command_file.seek(0)
              self._command_file.truncate(0)

              # Writes the command and its arguments to the command file
              cmd = self._handle_command(command)

              # Releases the lock to indicate the server that the command is
              # ready
              self._block_lock.release()

              # Waits for the answer to be written in the answer file
              # It prevents the block from re-acquiring the lock it just
              # released
              try:
                t = time()
                while not self._answer_file.tell():
                  sleep(0.00001)
                  if time() - t > 1:
                    raise TimeoutError
              except (KeyboardInterrupt, TimeoutError):
                raise_kbi = True
                self._block_lock.acquire(timeout=1)
                raise

              # When the lock is re-acquired, the server has written the answer
              # in the answer file
              if self._block_lock.acquire(timeout=1):
                # Reading the answer
                self._answer_file.seek(0)
                answer: List[bytes] = self._answer_file.read().split(b',')
                self._answer_file.seek(0)
                self._answer_file.truncate(0)

                # Making sure we're reading the answer corresponding to our
                # command
                if cmd != answer[0]:
                  raise KeyboardInterrupt

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
                raise TimeoutError("Couldn't acquire the lock in a reasonable "
                                   "delay !")

            except KeyboardInterrupt:
              # When interrupted, resend the same command to the server without
              # releasing the common lock
              raise_kbi = True
              release = False
              continue
            finally:
              # Releasing the locks if needed
              if release:
                try:
                  self._current_lock.release()
                except AssertionError:
                  pass
                try:
                  self._block_lock.release()
                except AssertionError:
                  pass

        else:
          raise TimeoutError("Couldn't acquire the lock in a reasonable "
                             "delay !")

      except KeyboardInterrupt:
        # Still try to send the command despite the KeyboardInterrupt
        # This way the block can stop properly
        continue
      finally:
        # Raise the KeyboardInterrupt that was postponed if needed
        if raise_kbi:
          raise KeyboardInterrupt

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
      self._set_bitmode(0xFF, ft232h.BitMode.MPSSE)
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
        buf = self._send_server(['ctrl_transfer_in', Ftdi_req_in,
                                 ft232h_sio_req['read_eeprom'], 0, word_addr,
                                 2, self._usb_read_timeout])
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

  def close(self) -> None:
    """Closes the FTDI interface/port."""

    if self._send_server(['close?']):

      if self._send_server(['_ctx.handle']):
        try:
          self._set_bitmode(0, ft232h.BitMode.RESET)
          self._send_server(['release_interface', self._index - 1])
        except (IOError, ValueError, USBError):
          pass
        try:
          self._send_server(['attach_kernel_driver', self._index - 1])
        except (NotImplementedError, USBError):
          pass
      self._send_server(['dispose_resources'])

    self._send_server(['farewell'])
