# coding: utf-8

from multiprocessing import Process, RLock, Event, Value, get_start_method
import multiprocessing.synchronize
import multiprocessing.context
import multiprocessing.queues
from multiprocessing.sharedctypes import Synchronized
import signal
from _io import FileIO
from tempfile import TemporaryFile
from typing import Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import logging.handlers

from ..._global import OptionalModule
try:
  from usb.core import find, Device, USBTimeoutError
  from usb import util
except (FileNotFoundError, ModuleNotFoundError):
  find = OptionalModule('pyusb')
  Device = OptionalModule
  USBTimeoutError = OptionalModule('pyusb')
  util = OptionalModule('pyusb')

USBArgsType = tuple[int, multiprocessing.synchronize.RLock, FileIO,
                    FileIO, multiprocessing.synchronize.RLock,
                    Synchronized]


@dataclass
class BlockObjects:
  """This class stores all the objects specific to a single Block, in order to
  easily pass them to the :class:`USBServer` process.

  Such a class will be created for each :class:`~crappy.blocks.Block`
  registering with the :class:`USBServer`.
  """

  ser_num: str
  lock: multiprocessing.synchronize.RLock
  device: Device

  finished: bool = False


class USBServer(Process):
  """This class is a server managing communication with USB devices through the
  :mod:`pyusb` library.

  As :mod:`pyusb` is not process-safe in Python, running a server is the only
  option to allow multiple :class:`~crappy.blocks.Block` to use the library in
  parallel. This server simply sends the USB commands it receives to the USB
  devices, and returns back the answers. It features a quite complex
  architecture for managing the requests and properly starting up and shutting
  down.

  The server is a child of :obj:`multiprocessing.Process`.
  
  .. versionadded:: 1.5.2
  .. versionchanged:: 2.0.0 renamed from *Usb_server* to *USBServer*
  """

  initialized = False
  logger: Optional[logging.Logger] = None

  process: Optional[multiprocessing.context.Process] = None
  block_nr: int = 0
  devices: dict[str, Device] = dict()

  # Objects for synchronizing with the server
  stop_event: Optional[multiprocessing.synchronize.Event] = None
  current_block: Optional[Synchronized] = None
  command_file: Optional[FileIO] = None
  answer_file: Optional[FileIO] = None
  shared_lock: Optional[multiprocessing.synchronize.RLock] = None
  block_dict: dict[int, BlockObjects] = dict()

  def __init__(self,
               current_block: Synchronized,
               command_file: FileIO,
               answer_file: FileIO,
               block_dict: dict[int, BlockObjects],
               stop_event: multiprocessing.synchronize.Event,
               log_queue: multiprocessing.queues.Queue,
               log_level: Optional[int]) -> None:
    """Sets the arguments.

    Args:
      current_block: A :obj:`multiprocessing.Value` storing the index of
        the :class:`~crappy.blocks.Block` currently allowed to communicate with
        the server.
      command_file: The handle to a file where the USB commands to send will be
        written.
      answer_file: The handle to a file where to write the answers from the USB
        devices.
      block_dict: A :obj:`dict` indicating for each index which
        :class:`~crappy.blocks.Block` it corresponds to.
      stop_event: A :obj:`multiprocessing.Event` indicating the server when it
        should stop running.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to
        the main :obj:`~logging.Logger`, only used in Windows.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
    """

    super().__init__(name=f'crappy.{type(self).__name__}')

    # Objects for synchronizing with the server
    self._current_block = current_block
    self._command_file = command_file
    self._answer_file = answer_file
    self._block_dict = block_dict
    self._stop_event = stop_event

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level

    # Keeping a track of the number of connected blocks for each FT232H
    self._dev_count = dict()
    for ser_num in set(block.ser_num for block in self._block_dict.values()):
      self._dev_count[ser_num] = sum(
        1 for _ in (block for block in self._block_dict.values()
                    if block.ser_num == ser_num))

  @classmethod
  def register(cls, ser_num: Optional[str] = None) -> USBArgsType:
    """Allows a :class:`~crappy.blocks.Block` to register for communicating
    with the server. This Block is then given the necessary information for
    communication.

    Args:
      ser_num: The serial number of the FT232H to communicate with, as a
        :obj:`str`.

    Returns:
      A :obj:`tuple` containing the necessary information for other objects to
      communicate with the server. This information is for example given as
      arguments to :class:`~crappy.tool.ft232h.FT232HServer` objects.
    
    .. versionadded:: 2.0.0
    """

    # Initializing the synchronization objects
    if not cls.initialized:
      cls._initialize()

    # Assigning an index to the calling Block
    cls.block_nr += 1
    index = cls.block_nr

    # If there's only one device connected and the serial number is not
    # specified, using the connected device
    if len(cls.devices) == 1 and (ser_num is None or ser_num == ""):
      device = list(cls.devices.values())[0]

    # Otherwise, making sure the serial number is one of the connected devices
    else:
      try:
        device = cls.devices[ser_num]
      except KeyError:
        raise IOError(f"No FT232H detected with serial number {ser_num} !")

    # Storing the relevant attributes for the calling Block
    lock = RLock()
    cls.block_dict[index] = BlockObjects(ser_num=ser_num,
                                         lock=lock,
                                         device=device)

    return (index, lock, cls.command_file, cls.answer_file, cls.shared_lock,
            cls.current_block)

  @classmethod
  def start_server(cls,
                   log_queue: multiprocessing.queues.Queue,
                   log_level: int) -> None:
    """Initializes and starts the USB server Process.

    Args:
      log_queue: The :obj:`multiprocessing.Queue` carrying the log messages 
        from the server Process to Crappy's centralized log handler. Only used 
        in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
    """

    cls.process = cls(current_block=cls.current_block,
                      command_file=cls.command_file,
                      answer_file=cls.answer_file,
                      block_dict=cls.block_dict,
                      stop_event=cls.stop_event,
                      log_queue=log_queue,
                      log_level=log_level)
    cls.process.start()

  @classmethod
  def stop_server(cls) -> None:
    """If the server was started, tries to stop it gently and if not successful
    terminates it.

    .. versionadded:: 2.0.0
    """

    if cls.process is not None:
      cls.stop_event.set()
      cls.log(logging.INFO, "Stop event set, waiting for the USB server to "
                            "finish")
      cls.process.join(0.2)

      if cls.process.is_alive():
        cls.log(logging.WARNING, "The USB server process did not stop "
                                 "correctly, killing it !")
        cls.process.terminate()

  @classmethod
  def _initialize(cls) -> None:
    """Sets the synchronization attributes and detects all the connected
    FT232H devices."""

    cls.devices = cls._get_devices()

    cls.stop_event = Event()
    cls.current_block = Value('B')
    cls.command_file = TemporaryFile(buffering=0)
    cls.answer_file = TemporaryFile(buffering=0)
    cls.shared_lock = RLock()

    cls.initialized = True

  @classmethod
  def log(cls, level: int, msg: str) -> None:
    """Wrapper for recording log messages.

    Also instantiates the :obj:`~logging.Logger` on the first message.

    Args:
      level: The logging level of the message, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.

    .. versionadded:: 2.0.0
    """

    if cls.logger is None:
      cls.logger = logging.getLogger(f'crappy.{cls.__name__}')

    cls.logger.log(level, msg)

  @staticmethod
  @contextmanager
  def acquire_timeout(lock: multiprocessing.synchronize.RLock,
                      timeout: float) -> bool:
    """Short context manager for acquiring a :obj:`multiprocessing.Lock` with a 
    specified timeout.

    Args:
      lock: The Lock to acquire.
      timeout: The timeout for acquiring the Lock, as a :obj:`float`.

    Returns:
      :obj:`True` if the Lock was successfully acquired, :obj:`False`
      otherwise.

    .. versionadded:: 2.0.0
    """

    ret = False
    try:
      ret = lock.acquire(timeout=timeout)
      yield ret
    finally:
      if ret:
        lock.release()

  @staticmethod
  def _get_devices() -> dict[str, Any]:
    """Detects all the connected FT232H devices and returns them as a 
    :obj:`dict`.

    Returns:
      A :obj:`dict` containing as keys the detected serial numbers and as
      values the handles in the :mod:`pyusb` module to the associated FT232H
      devices.
    """

    # Searching for the FT232H devices
    devices: list[Device] = list(find(find_all=True,
                                      idVendor=0x0403,
                                      idProduct=0x6014))
    if not devices:
      raise IOError("No FT232H connected !")

    dev_dict = {}

    # Storing the found devices
    for device in devices:
      try:
        dev_dict[device.serial_number] = device

      except ValueError:
        # If there's only one FT232H connected, it can lack a serial number
        if len(devices) == 1:
          dev_dict[''] = devices[0]
        # Otherwise, every FT232H must have a serial number
        else:
          raise ValueError('Please set a serial number for each FT232H ! It '
                           'can be done using the dedicated crappy tool')

    return dev_dict

  def run(self) -> None:
    """The main loop of the server.

    Waits for a :class:`~crappy.blocks.Block` to acquire control, reads its 
    command, sends it to the correct USB device, reads the answer from the USB 
    device and sends it back to the Block in control. Then, does the same with 
    the next Block getting control over the server.

    .. versionadded:: 2.0.0
    """

    self._set_logger()
    self._log(logging.INFO, "Logger configured")

    # Disabling the KeyboardInterrupt exceptions, to avoid disruptions
    self._log(logging.WARNING, "Disabling KeyboardInterrupt for the server !")
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while not self._stop_event.is_set():

      # Exiting if all the devices have been closed
      if all(dev.finished for dev in self._block_dict.values()):
        self._log(logging.INFO, "Server finished after all devices were "
                                "closed")
        break

      # A Block has acquired the lock and shared its index
      if index := self._current_block.value:
        # Reading the index and resetting the value
        self._current_block.value = 0
        self._log(logging.DEBUG, f"Block with index {index} now has control")

        # The Block has finished writing the command
        with self.acquire_timeout(self._block_dict[index].lock, 1) as acquired:
          if acquired:
            self._log(logging.DEBUG, f"Acquired lock of Block with index "
                                     f"{index}")
            # Reading the command
            self._command_file.seek(0)
            command = self._command_file.read()
            # Resetting the command file
            self._command_file.seek(0)
            self._command_file.truncate(0)

            self._log(logging.DEBUG, f"Received command {command}")

            try:
              # Sends the command to the device and returns the answer
              answer = self._send_command(command,
                                          self._block_dict[index].device,
                                          self._block_dict[index].ser_num,
                                          index)
            except (USBTimeoutError, TimeoutError):
              # Double-checking the timeout error
              answer = self._send_command(command,
                                          self._block_dict[index].device,
                                          self._block_dict[index].ser_num,
                                          index)

            self._log(logging.DEBUG,
                      f"Got answer {answer} from device with serial number "
                      f"{self._block_dict[index].ser_num}")

            # Resetting the answer file
            self._answer_file.seek(0)
            self._answer_file.truncate(0)
            # Writing the answer in the answer file
            try:
              self._answer_file.write(answer)
            except TypeError:
              # Sometimes for an unknown reason the answer is None
              raise IOError("Got an unexpected USB answer from an FT232H !")

          else:
            raise TimeoutError(
              f"Could not acquire the Block lock of Block with index {index} "
              f"driving the FT232H with serial number "
              f"{self._block_dict[index].ser_num} within 1s, aborting !")

    if self._stop_event.is_set():
      self._log(logging.INFO, "Server finished after stop event was set")

  def _send_command(self,
                    command: bytes,
                    device: Device,
                    serial_nr: str,
                    index: int) -> bytes:
    """Sends commands to a USB device and returns the answer.

    Args:
      command: The command to send to the device. The bytes are arranged in a
        specific way for each type of command.
      device: The :mod:`pyusb` Device to which the commands are sent.
      serial_nr: The serial number of the :mod:`pyusb` device.
      index: The index of the Block currently controlling the server.

    Returns:
      The index of the command, followed by the answer from the USB device if
      applicable.
    """

    command = command.split(b',')

    # Control transfer out
    if command[0] == b'00':
      return b','.join((b'00',
                        str(device.ctrl_transfer(int(command[1]),
                                                 int(command[2]),
                                                 int(command[3]),
                                                 int(command[4]),
                                                 command[5],
                                                 int(command[6]))).encode()))
    # Control transfer in
    elif command[0] == b'01':
      return b','.join((b'01',
                        bytes(device.ctrl_transfer(*[int(arg) for arg in
                                                     command[1:]]))))
    # Write operation
    elif command[0] == b'02':
      return b','.join((b'02',
                        str(device.write(int(command[1]), command[2],
                                         int(command[3]))).encode()))
    # Read operation
    elif command[0] == b'03':
      return b','.join((b'03',
                        bytes(device.read(*[int(arg) for arg
                                            in command[1:]]))))
    # Checks whether the kernel driver is active
    # It doesn't actually interact with the device
    elif command[0] == b'04':
      return b','.join((b'04',
                        b'1' if device.is_kernel_driver_active(int(command[1]))
                        else b'0'))
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == b'05':
      device.detach_kernel_driver(int(command[1]))
      return b'05,'
    # Sets the device configuration
    elif command[0] == b'06':
      device.set_configuration()
      return b'06,'
    # Custom command getting information from the current configuration
    elif command[0] == b'07':
      info = self._return_config_info(device)
      return b','.join((b'07',
                        str(info[0]).encode(), str(info[1]).encode(),
                        str(info[2]).encode(), str(info[3]).encode()))
    # When a block is leaving, if it's the last one associated with a given
    # ft232h then it should release the internal resources
    # It doesn't actually interact with the device
    elif command[0] == b'08':
      self._dev_count[serial_nr] -= 1
      return b','.join((b'08',
                        b'1' if not self._dev_count[serial_nr] else b'0'))

    # Checks whether the internal resources have been released or not
    # It doesn't actually interact with the device
    elif command[0] == b'09':
      return b','.join((b'09', b'1' if device._ctx.handle else b'0'))
    # Releases the USB interface
    # It doesn't actually interact with the device
    elif command[0] == b'10':
      util.release_interface(device, int(command[1]))
      return b'10,'
    # Detaches the kernel driver
    # It doesn't actually interact with the device
    elif command[0] == b'11':
      device.attach_kernel_driver(int(command[1]))
      return b'11,'
    # Releases all the resources used by :mod:`pyusb` for a given device
    # It doesn't actually interact with the device
    elif command[0] == b'12':
      util.dispose_resources(device)
      return b'12,'
    # Registers a block as gone
    # It doesn't actually interact with the device
    elif command[0] == b'13':
      self._block_dict[index].finished = True
      return b'13,'

  def _set_logger(self) -> None:
    """Instantiates and sets up the :obj:`~logging.Logger` for recording log
    messages."""

    logger = logging.getLogger(self.name)

    # Disabling logging if requested
    if self._log_level is not None:
      logger.setLevel(self._log_level)
    else:
      logging.disable()

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn" and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger

  def _log(self, level: int, msg: str) -> None:
    """Wrapper for recording log messages.

    Args:
      level: The logging level of the message, as an :obj:`int`.
      msg: The message to lof, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(level, msg)

  @staticmethod
  def _return_config_info(device) -> tuple[int, int, int, int]:
    """Returns some configuration information from a USB object.

    Args:
      device: A :obj:`usb.core.Device`

    Returns:
      The index, in endpoint, out endpoint and maximum packet size of a USB
      device.
    """

    interface = device.get_active_configuration()[(0, 0)]
    index = interface.bInterfaceNumber + 1
    in_ep, out_ep = sorted([ep.bEndpointAddress for ep in interface])[:2]
    max_packet_size = interface[0].wMaxPacketSize
    return index, in_ep, out_ep, max_packet_size
