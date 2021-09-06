# coding: utf-8

from time import time, sleep
from multiprocessing import Process, Queue, Event, Condition
from multiprocessing.managers import SyncManager, Namespace
from signal import signal, SIGINT, SIG_IGN
from queue import Empty
from .._global import OptionalModule
try:
  from usb.core import find
  from usb import util
except (FileNotFoundError, ModuleNotFoundError):
  find = OptionalModule('pyusb')
  util = OptionalModule('pyusb')


def _int_handler() -> None:
  """Method for handling the :obj:`signal.SIGINT` signal in the
  :obj:`multiprocessing.managers.SyncManager` process."""

  signal(SIGINT, SIG_IGN)


def return_config_info(device) -> tuple:
  """Returns some configuration information from a USB object.

  It is meant to send back only pickable data to the FT232H object.

  Args:
    device: A :obj:`usb.core.Device`

  Returns:
    The index, in endpoint, out endpoint and maximum packet size of a USB device
  """

  interface = device.get_active_configuration()[(0, 0)]
  index = interface.bInterfaceNumber + 1
  in_ep, out_ep = sorted([ep.bEndpointAddress for ep in interface])[:2]
  max_packet_size = interface[0].wMaxPacketSize
  return index, in_ep, out_ep, max_packet_size


class Usb_server:
  """Class for starting a server controlling communication with the FT232H
  devices.

  The :ref:`In / Out` objects wishing to communicate through an FT232H inherit
  from this class.
  """

  def __init__(self, serial_nr: str, backend: str) -> None:
    """Simply receives the attributes from the :ref:`In / Out` object.

    Args:
      serial_nr (:obj:`int`): The serial number of the FT232H to use.
      backend (:obj:`str`): The server won't be started if the chosen backend is
        not ``'ft232h'``.
    """

    self._serial_nr = serial_nr
    self._backend = backend

  def start_server(self) -> tuple:
    """Starts the server for communicating with the FT232H devices.

    If the server is already started, doesn't start it twice. Then initializes
    the connection with the server and receives a block number.

    Returns:
      The different :mod:`multiprocessing` objects needed as arguments by the
      FT232H in order to run properly.
    """

    if self._backend == 'ft232h':
      if not hasattr(Usb_server, 'server'):
        # Finding all connected FT232H
        devices = list(find(find_all=True,
                            idVendor=0x0403,
                            idProduct=0x6014))
        if not devices:
          raise IOError("No FT232H connected")

        dev_dict = {}

        # Collecting all the serial numbers of the connected FT232H
        if len(devices) == 0:
          raise IOError("No FT232H connected")
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

        Usb_server._queue = Queue()  # One unique queue for all the InOuts
        Usb_server.stop_event = Event()  # Event for stopping the server
        manager = SyncManager()
        manager.start(_int_handler)
        Usb_server.namespace = manager.Namespace()
        Usb_server.command_event = Event()
        Usb_server.answer_event = Event()
        Usb_server.next_process = Event()
        Usb_server.done_event = Event()
        Usb_server.namespace.current_block = None
        # Starting the server process
        Usb_server.server = Process(target=self._run,
                                    kwargs={'queue': Usb_server._queue,
                                            'stop_event':
                                                Usb_server.stop_event,
                                            'dev_dict': dev_dict,
                                            'namespace':
                                                Usb_server.namespace,
                                            'command_event':
                                                Usb_server.command_event,
                                            'answer_event':
                                                Usb_server.answer_event,
                                            'next_process':
                                                Usb_server.next_process,
                                            'done_event':
                                                Usb_server.done_event},
                                    daemon=True)
        Usb_server.server.start()

      # Sending the pipes handles to the server
      Usb_server._queue.put_nowait({'serial_nr': self._serial_nr})
      # Receiving the block number from the server
      if Usb_server.next_process.wait(timeout=5):
        block_number = Usb_server.namespace.current_block
        Usb_server.next_process.clear()
        if isinstance(block_number, Exception):
          raise block_number
        self.block_number = block_number
      else:
        raise TimeoutError('The USB server took too long to reply')

      return Usb_server._queue, self.block_number, Usb_server.namespace, \
          Usb_server.command_event, Usb_server.answer_event, \
          Usb_server.next_process, Usb_server.done_event

    return None, None, None, None, None, None

  def __del__(self) -> None:
    if hasattr(Usb_server,
               'stop_event') and not Usb_server.stop_event.is_set():
      Usb_server.stop_event.set()
      sleep(1)
    if hasattr(Usb_server, 'server') and Usb_server.server.is_alive():
      Usb_server.server.kill()

  @staticmethod
  def _run(queue: Queue, stop_event: Event, dev_dict: dict,
           namespace: Namespace, command_event: Event, answer_event: Event,
           next_process: Event, done_event: Event) -> None:
    """The loop of the FT232H server.

    First registers the new blocks trying to connect and assigns them a number.
    Then grants control to 1 block at a time, executes its commands and sends
    back the corresponding answers. Many securities are implemented to ensure
    that the right message is sent at the right time.

    Args:
      queue: The blocks put their number in a queue that determines in which
        order they can control the server.
      stop_event: An event that can stop the server when set.
      dev_dict: A dictionary containing all the connected devices.
      namespace: A Namespace allowing to share both the commands and the
        answers.
      command_event: An event, set when a block has written a command in the
        Namespace.
      answer_event: An event, set when the server has written an answer in the
        Namespace.
      next_process: An event, set when the server is ready to switch to the next
        block
      done_event: An event, set when a block is done controlling the server.
    """

    try:
      i = 0
      t = time()
      condition = Condition()
      condition.acquire()
      dev_count = {serial_nr: 0 for serial_nr in dev_dict}
      blocks = {}
      block = None
      while True:
        try:
          # Stopping the process when asked to
          if stop_event.is_set() or \
                (all(block['left'] for block in blocks.values()) and blocks):
            print("STOP EVENT")
            break

          # Queue for acquiring the control over the server
          if block is None:
            try:
              block = queue.get(block=True, timeout=1)
              t = time()
              if isinstance(block, int) and blocks[block]['left']:
                block = None
                continue
            except Empty:
              block = None
        except KeyboardInterrupt:
          continue

        # One block wants control over the server
        if block is not None:
          # Assigning a number to each block at first interaction
          if isinstance(block, dict):
            try:
              blocks[i] = block
              blocks[i]['left'] = False
              # Checking if the serial number associated with the block is valid
              if blocks[i]['serial_nr'] not in dev_dict:
                namespace.current_block = ValueError(
                  "FT232H with specified serial number ({}) is not "
                  "connected".format(blocks[i]['serial_nr']))
                next_process.set()
                break
              namespace.current_block = i
              setattr(namespace, 'command_event' + str(i), False)
              setattr(namespace, 'answer_event' + str(i), False)
              t = time()
              next_process.set()
            except KeyboardInterrupt:
              continue
            try:
              block = None
              dev_count[blocks[i]['serial_nr']] += 1
              i += 1
              continue
            except KeyboardInterrupt:
              block = None
              dev_count[blocks[i]['serial_nr']] += 1
              i += 1
              continue

          try:
            namespace.current_block = block
            setattr(namespace, 'answer' + str(block), None)
            setattr(namespace, 'answer' + str(block) + "'", None)
            if done_event.wait(timeout=1):
              next_process.set()
              done_event.clear()
            elif time() - t < 1:
              continue
            else:
              setattr(namespace, 'answer' + str(block), TimeoutError(
                "Previous process took too long to release control"))
            # Retrieving the device object
            dev = dev_dict[blocks[block]['serial_nr']]

            # Executes commands and sends the output until told to stop
          except KeyboardInterrupt:
            namespace.current_block = block
            if done_event.wait(timeout=1):
              next_process.set()
              done_event.clear()
            elif time() - t < 1:
              setattr(namespace, 'answer' + str(block), TimeoutError(
                "Previous process took too long to release control"))
            else:
              continue
            # Retrieving the device object
            dev = dev_dict[blocks[block]['serial_nr']]
          while True:
            try:
              if command_event.wait(timeout=3):
                # In case the process was KeyboardInterrupt-ed, doesn't try to
                # receive a new command
                command = getattr(namespace, 'command' + str(block))
                t = time()
                if command is None:
                  continue
                command_event.clear()
                # Exiting the while loop if no more command to send
                if command == 'stop':
                  setattr(namespace, 'answer' + str(block), 'ok')
                  setattr(namespace, 'answer' + str(block) + "'", 'ok')
                  setattr(namespace, 'command' + str(block), None)
                  answer_event.set()
                  break
                if command == 'farewell':
                  setattr(namespace, 'answer' + str(block), 'ok')
                  setattr(namespace, 'answer' + str(block) + "'", 'ok')
                  setattr(namespace, 'command' + str(block), None)
                  answer_event.set()
                  blocks[block]['left'] = True
                  break
                # Specific type for device attributes
                elif isinstance(command, str):
                  # Cannot send handle object in pipes, so here's a workaround
                  if command == '_ctx.handle':
                    out = bool(dev._ctx.handle)
                    # Counting the remaining blocks connected to a device
                  elif command == 'close?':
                    out = dev_count[blocks[block]['serial_nr']] <= 1
                    dev_count[blocks[block]['serial_nr']] -= 1
                  else:
                    command = command.split('.')
                    out = dev
                    while command:
                      out = getattr(out, command[0])
                      command = command[1:]
                elif isinstance(command, list):
                  # specific syntax for usb.util methods
                  if not isinstance(command[0], str):
                    command = [item if item != 'dev' else dev
                               for item in command[1:]]
                    out = getattr(util, command[0])(*command[1:])
                  # Cannot send usb configuration objects in pipes, so here's a
                  # workaround
                  elif command[0] == 'get_active_configuration':
                    out = return_config_info(dev)
                  # Cannot set configuration twice on a same device
                  elif command[0] == 'set_configuration':
                    try:
                      out = getattr(dev, command[0])(*command[1:])
                    except IOError:
                      out = None
                  # Specific syntax for device methods
                  else:
                    out = getattr(dev, command[0])(*command[1:])
                # If command is not str or list, there's an issue
                else:
                  out = TypeError("Wrong type for the command")
                  setattr(namespace, 'answer' + str(block), out)
                  setattr(namespace, 'answer' + str(block) + "'", out)
                  answer_event.set()
                  break
                if command_event.is_set():
                  continue
                if out is None:
                  out = ''
                setattr(namespace, 'answer' + str(block), out)
                setattr(namespace, 'answer' + str(block) + "'", out)
                answer_event.set()
              # If there's no communication during 1s maybe the block is down
              elif stop_event.is_set() or \
                  (all(block['left'] for key, block in blocks.items())
                   and blocks):
                break
              else:
                if command_event.wait(timeout=1):
                  continue
                elif time() - t < 1:
                  continue
                setattr(namespace, 'answer' + str(block), None)
                setattr(namespace, 'answer' + str(block) + "'", None)
                answer_event.clear()
                done_event.set()
                queue.put_nowait(block)
                break

            except KeyboardInterrupt:
              command_event.clear()
              answer_event.clear()
              continue
        try:
          block = None
        except KeyboardInterrupt:
          block = None
    except (RuntimeError, ConnectionResetError,
            BrokenPipeError, AssertionError):
      pass
