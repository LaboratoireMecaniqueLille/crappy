# coding: utf-8

from multiprocessing import Process, Pipe, RLock
from multiprocessing.connection import Connection
import multiprocessing.synchronize
from _io import FileIO
from tempfile import TemporaryFile
from typing import List, Dict
from .._global import OptionalModule
try:
  from usb.core import find, Device, USBTimeoutError
  from usb import util
except (FileNotFoundError, ModuleNotFoundError):
  find = OptionalModule('pyusb')
  Device = OptionalModule('pyusb')
  USBTimeoutError = OptionalModule('pyusb')
  util = OptionalModule('pyusb')


class Server_process(Process):
  """Process actually communicating with the FT232H device.

  It receives the commands, sends them to the device, and returns the answer.
  This architecture is necessary as :mod:`pyusb` doesn't support
  multiprocessing.
  """

  def __init__(self,
               new_block_recv: Connection,
               current_file: FileIO,
               command_file: FileIO,
               answer_file: FileIO,
               lock_pool: List[multiprocessing.synchronize.RLock],
               current_lock: multiprocessing.synchronize.RLock,
               dev_dict: Dict[str, Device]):
    """Simply passes the args as instance attributes.

    Args:
      new_block_recv: A pipe connection through which new blocks send
        information.
      current_file: A temporary file in which the index of the block currently
        communicating with the server is written.
      command_file: A temporary file containing the command the server has to
        send to the ft232h.
      answer_file: A temporary file containing the answer from the device after
        a command was sent.
      lock_pool: A :obj:`list` of RLocks, with each one affected to a different
        block. They indicate to the server that a command is ready, or to the
        block that an answer is ready.
      current_lock: A unique RLock that determines which block has control over
        the server. The different blocks all try to acquire this lock.
      dev_dict: A :obj:`dict` whose keys are the serial numbers of the
        connected ft232h and values are the associated :mod:`pyusb` Device
        objects.
    """

    super().__init__()

    self.new_block_recv = new_block_recv
    self.current_file = current_file
    self.command_file = command_file
    self.answer_file = answer_file
    self.lock_pool = lock_pool
    self.current_lock = current_lock
    self.dev_dict = dev_dict

  def _send_command(self,
                    command: bytes,
                    device: Device,
                    serial_nr: str,
                    current_block: int) -> bytes:
    """Sends commands to the USB devices and returns the answer.

    Args:
      command: The command to send to the device. The bytes are arranged in a
        specific way for each type of command.
      device: The :mod:`pyusb` Device to which the commands are sent.
      serial_nr: The serial number of the :mod:`pyusb` device.
      current_block: The index of the block currently controlling the server.

    Returns:
      The number of the command, followed by the answer from the USB device if
      any.
    """

    command = command.split(b',')

    # Control transfer out
    if command[0] == b'00':
      return b'00,' + str(device.ctrl_transfer(int(command[1]),
                                               int(command[2]),
                                               int(command[3]),
                                               int(command[4]),
                                               command[5],
                                               int(command[6]))).encode()
    # Control transfer in
    elif command[0] == b'01':
      return b'01,' + bytes(device.ctrl_transfer(*[int(arg) for arg in
                                                   command[1:]]))
    # Write operation
    elif command[0] == b'02':
      return b'02,' + str(device.write(int(command[1]), command[2],
                                       int(command[3]))).encode()
    # Read operation
    elif command[0] == b'03':
        return b'03,' + bytes(device.read(*[int(arg) for arg in command[1:]]))
    # Checks whether the kernel driver is active
    # It doesn't actually interact with the device
    elif command[0] == b'04':
      return b'04,' + (b'1' if device.is_kernel_driver_active(int(command[1]))
                       else b'0')
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
      return b'07,' + str(info[0]).encode() + b',' + str(info[1]).encode() + \
             b',' + str(info[2]).encode() + b',' + str(info[3]).encode()
    # When a block is leaving, if it's the last one associated with a given
    # ft232h then it should release the internal resources
    # It doesn't actually interact with the device
    elif command[0] == b'08':
      self.dev_count[serial_nr] -= 1
      return b'08,' + (b'1' if not self.dev_count[serial_nr] else b'0')
    # Checks whether the internal resources have been released or not
    # It doesn't actually interact with the device
    elif command[0] == b'09':
      return b'09,' + (b'1' if device._ctx.handle else b'0')
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
      self.left[current_block] = True
      return b'13,'

  def run(self) -> None:
    """Main loop of the server.

    It runs an infinite loop for receiving the commands and sending back the
    answers to the blocks.
    """

    block_count = 0  # The count of blocks registered with the server
    num_to_dev = {}  # Associates each block index to a ft232h device
    num_to_ser = {}  # Associates each block index to a serial number
    self.left = {}  # For each block index, tells whether the block has left
    # Counts the number of blocks controlling a given device
    self.dev_count = {serial_nr: 0 for serial_nr in self.dev_dict}
    while True:
      try:
        # If all the blocks have left, stop the server
        if self.left and all(val for val in self.left.values()):
          break

        # A new block wants to register
        if self.new_block_recv.poll():
          try:
            # It sends a Connection object
            temp_pipe = self.new_block_recv.recv()
            if temp_pipe.poll(timeout=1):
              # The serial number of the ft232h the block wants to control is
              # sent
              serial_nr = temp_pipe.recv()
              # The different dicts are updated accordingly
              num_to_dev[block_count] = self.dev_dict[serial_nr]
              num_to_ser[block_count] = serial_nr
              self.dev_count[serial_nr] += 1
              self.left[block_count] = False
            else:
              raise TimeoutError("A block took too long to send the serial nr")
            # Sending back the index the block was assigned
            temp_pipe.send(block_count)
            block_count += 1
          except KeyboardInterrupt:
            # The program was interrupted during __init__, no choice but to
            # stop abruptly
            break

        # A block has acquired the lock and wrote its index in the file
        if self.current_file.tell() > 0:
          self.current_file.seek(0)
          # Reads the index of the block currently in control
          current_block = int(self.current_file.read())
          self.current_file.seek(0)
          self.current_file.truncate(0)

          # The block has finished writing the command
          if self.lock_pool[current_block].acquire(timeout=1):
            try:
              # Reads the command
              self.command_file.seek(0)
              command = self.command_file.read()
              self.command_file.seek(0)
              self.command_file.truncate(0)

              try:
                # Sends the command to the device and returns the answer
                answer = self._send_command(command,
                                            num_to_dev[current_block],
                                            num_to_ser[current_block],
                                            current_block)
              except (USBTimeoutError, TimeoutError):
                # Double-checking the timeout error
                answer = self._send_command(command,
                                            num_to_dev[current_block],
                                            num_to_ser[current_block],
                                            current_block)

              # Writing the answer in the file
              self.answer_file.seek(0)
              self.answer_file.truncate(0)
              try:
                self.answer_file.write(answer)
              except TypeError:
                # Sometimes for an unknown reason the answer is None
                raise KeyboardInterrupt

            except KeyboardInterrupt:
              # If the command wasn't sent it's no big deal, the block will
              # send it again anyway
              pass
            finally:
              # Releasing the lock so that the block can go on
              self.lock_pool[current_block].release()

      except KeyboardInterrupt:
        # The server should never raise exceptions, as it must keep running to
        # allow the blocks to finish properly
        continue

  @staticmethod
  def _return_config_info(device) -> tuple:
    """Returns some configuration information from a USB object.

    Args:
      device: A :obj:`usb.core.Device`

    Returns:
      The index, in endpoint, out endpoint and maximum packet size of a USB
      device
    """

    interface = device.get_active_configuration()[(0, 0)]
    index = interface.bInterfaceNumber + 1
    in_ep, out_ep = sorted([ep.bEndpointAddress for ep in interface])[:2]
    max_packet_size = interface[0].wMaxPacketSize
    return index, in_ep, out_ep, max_packet_size


class Usb_server:
  """Class for starting a server controlling communication with the
  :ref:`FT232H` devices.

  The :ref:`In / Out` objects wishing to communicate through an :ref:`FT232H`
  inherit from this class.

  Note:
    There is a limitation to 10 blocks accessing ft232h devices from the same
    machine in Crappy. This limit can be increased at will, but it is necessary
    to change the code of this class and build Crappy yourself.
  """

  def __init__(self, serial_nr: str, backend: str) -> None:
    """Simply receives the attributes from the :ref:`In / Out` object.

    Args:
      serial_nr (:obj:`int`): The serial number of the :ref:`FT232H` to use.
      backend (:obj:`str`): The server won't be started if the chosen backend
        is not ``'ft232h'``.
    """

    self._serial_nr = serial_nr
    self._backend = backend

  def start_server(self) -> tuple:
    """Starts the server for communicating with the :ref:`FT232H` devices.

    If the server is already started, doesn't start it twice. Then initializes
    the connection with the server and receives a block number.

    Returns:
      The different :mod:`multiprocessing` objects needed as arguments by the
      :ref:`FT232H` in order to run properly.
    """

    if self._backend == 'ft232h':
      # The server should only be started once
      if not hasattr(Usb_server, 'server'):
        # Finding all connected FT232H
        devices: List[Device] = list(find(find_all=True,
                                          idVendor=0x0403,
                                          idProduct=0x6014))
        if not devices:
          raise IOError("No FT232H connected")

        dev_dict = {}

        # Collecting all the serial numbers of the connected FT232H
        if len(devices) == 0:
          raise IOError("No FT232H connected")
        # If only one FT232H is connected then it is acceptable not to give a
        # serial number
        elif len(devices) == 1:
          if self._serial_nr == '':
            dev_dict[''] = devices[0]
          else:
            dev_dict[devices[0].serial_number] = devices[0]
        else:
          for device in devices:
            try:
              dev_dict[device.serial_number] = device
            except ValueError:
              if len(devices) > 1:
                raise ValueError('Please set a serial number for each FT232H '
                                 'using the corresponding crappy tool')
              else:
                dev_dict[''] = device

        # This pipe is used by each new block to send a Connection object
        # Through this connection the server receives the information of the
        # new block
        Usb_server.new_block_send, new_block_recv = Pipe()
        # In this file will be written the number of the block currently owning
        # the lock
        Usb_server.current_file = TemporaryFile(buffering=0)
        # In this file will be written the command from the block to the server
        Usb_server.command_file = TemporaryFile(buffering=0)
        # In this file will be written the answer from the server to the block
        Usb_server.answer_file = TemporaryFile(buffering=0)
        # Each new block is assigned one of these locks
        # It is used by the block to indicate the server that the command is
        # ready, and by the server to indicate the block that the answer is
        # ready
        Usb_server.lock_pool = [RLock() for _ in range(10)]
        # The block owning this lock is the only one that can communicate with
        # the server
        Usb_server.current_lock = RLock()

        # Starting the server process
        Usb_server.server = Server_process(
          new_block_recv, Usb_server.current_file,
          Usb_server.command_file,
          Usb_server.answer_file, Usb_server.lock_pool,
          Usb_server.current_lock, dev_dict)
        Usb_server.server.start()

      # Sending the server the serial number of the ft232h to use
      temp_pipe_block, temp_pipe_server = Pipe()
      Usb_server.new_block_send.send(temp_pipe_server)
      temp_pipe_block.send(self._serial_nr)
      if temp_pipe_block.poll(timeout=1):
        # Receiving the block number from the server
        self.block_number = temp_pipe_block.recv()
        self.block_lock = Usb_server.lock_pool[self.block_number]
      else:
        raise TimeoutError('The USB server took too long to reply')

      # Transmitting the necessary information to the InOut or Actuator object
      return Usb_server.current_file, self.block_number, \
          Usb_server.command_file, Usb_server.answer_file, \
          self.block_lock, Usb_server.current_lock

    return None, None, None, None, None, None

  def __del__(self) -> None:
    """Stops the server upon deletion of the :ref:`In / Out` object."""

    if hasattr(Usb_server, 'server') and Usb_server.server.is_alive():
      Usb_server.server.kill()
