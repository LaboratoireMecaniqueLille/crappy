# coding: utf-8

from time import time, sleep
from multiprocessing import Process, Queue, Event
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


def _return_config_info(device) -> tuple:
  """Returns some configuration information from a USB object.

  It is meant to send back only pickable data to the :ref:`FT232H` object.

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
  """Class for starting a server controlling communication with the
  :ref:`FT232H` devices.

  The :ref:`In / Out` objects wishing to communicate through an :ref:`FT232H`
  inherit from this class.
  """

  def __init__(self, serial_nr: str, backend: str) -> None:
    """Simply receives the attributes from the :ref:`In / Out` object.

    Args:
      serial_nr (:obj:`int`): The serial number of the :ref:`FT232H` to use.
      backend (:obj:`str`): The server won't be started if the chosen backend is
        not ``'ft232h'``.
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
        devices = list(find(find_all=True,
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

        Usb_server._queue = Queue()  # One unique queue for all the InOuts
        Usb_server.stop_event = Event()  # Event for stopping the server
        manager = SyncManager()  # Manager for sending commands and answers
        manager.start(_int_handler)  # For ignoring the SIGINT signal
        Usb_server.namespace = manager.Namespace()
        # Different events for synchronization of the server and the InOuts
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

      # Sending the list of connected devices to the server
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
    """Stops the server upon deletion of the :ref:`In / Out` object."""

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
    """The loop of the USB server.

    First registers the new blocks trying to connect and assigns them a number.
    Then grants control to one block at a time, executes its commands and sends
    back the corresponding answers. Many securities are implemented to ensure
    that the right message is sent at the right time.

    Args:
      queue (:obj:`multiprocessing.Queue`): The blocks put their number in a
        queue that determines in which order they can control the server.
      stop_event (:obj:`multiprocessing.Event`): An event that can stop the
        server when set.
      dev_dict (:obj:`dict`): A dictionary containing all the connected devices
        and their serial numbers.
      namespace (:obj:`multiprocessing.managers.Namespace`): A Namespace
        allowing to share both the commands and the answers with the
        :ref:`In / Out`.
      command_event (:obj:`multiprocessing.Event`): An event, set  when a block
        has written a command in the Namespace.
      answer_event (:obj:`multiprocessing.Event`): An event, set when the server
        has written an answer in the Namespace.
      next_process (:obj:`multiprocessing.Event`): An event, set when the server
        is ready to switch to the next block
      done_event (:obj:`multiprocessing.Event`): An event, set when a block is
        done controlling the server.
    """

    try:
      i = 0  # The block counter
      # The timestamp of the last interaction with the blocks is constantly
      # being saved because the multiprocessing objects timeouts are buggy
      t = time()
      # A counter recording how many blocks are controlling a same FT232H
      dev_count = {serial_nr: 0 for serial_nr in dev_dict}
      blocks = {}
      block = None
      while True:
        # This first loop grants the blocks control over the USB server
        try:
          # Stopping the process either when the stop_event is set or when all
          # the registered blocks have sent a 'farewell' command
          if stop_event.is_set() or \
                (all(block['left'] for block in blocks.values()) and blocks):
            break

          # Getting the next block that will control the server from the queue
          if block is None:
            try:
              block = queue.get(block=True, timeout=1)
              t = time()
              # It may happen that a block in the queue has actually already
              # left
              if isinstance(block, int) and blocks[block]['left']:
                block = None
                continue
            except Empty:
              block = None
        except KeyboardInterrupt:
          continue

        # One block wants control over the server
        if block is not None:
          # At first interaction with the server the block sends a dict
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
              # Assigning a number to the new block
              namespace.current_block = i
              t = time()
              # Ready to switch to the next process
              next_process.set()
            except KeyboardInterrupt:
              continue
            try:
              # Increasing the counters by 1
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
            # Resetting the commands and answers to make sure old ones cannot be
            # used
            setattr(namespace, 'answer' + str(block), None)
            setattr(namespace, 'answer' + str(block) + "'", None)
            # Can only switch to the next block if the previous one signals
            # itself as done
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

          except KeyboardInterrupt:
            namespace.current_block = block
            # Resetting the commands and answers to make sure old ones cannot be
            # used
            setattr(namespace, 'answer' + str(block), None)
            setattr(namespace, 'answer' + str(block) + "'", None)
            # Can only switch to the next block if the previous one signals
            # itself as done
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
          while True:
            # This second loop receives commands and sends back answers
            try:
              if command_event.wait(timeout=3):
                # Getting the command from the block
                command = getattr(namespace, 'command' + str(block))
                t = time()
                if command is None:
                  # The command shouldn't be None
                  continue
                command_event.clear()
                if command == 'stop':
                  # The block wants to release control
                  setattr(namespace, 'answer' + str(block), 'ok')
                  setattr(namespace, 'answer' + str(block) + "'", 'ok')
                  setattr(namespace, 'command' + str(block), None)
                  answer_event.set()
                  break
                if command == 'farewell':
                  # the block wants to leave forever
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
                  elif command == 'close?':
                    # The block wants to know if it should close the FT232H
                    # It can do so if it is the last block in control of it to
                    # leave
                    out = dev_count[blocks[block]['serial_nr']] <= 1
                    dev_count[blocks[block]['serial_nr']] -= 1
                  else:
                    command = command.split('.')
                    out = dev
                    while command:
                      out = getattr(out, command[0])
                      command = command[1:]
                elif isinstance(command, list):
                  if not isinstance(command[0], str):
                    # Specific syntax for usb.util methods
                    # Syntax for telling the server to use the dev object
                    command = [item if item != 'dev' else dev
                               for item in command[1:]]
                    out = getattr(util, command[0])(*command[1:])
                  elif command[0] == 'get_active_configuration':
                    # Cannot send usb configuration objects in pipes, so here's
                    # a workaround
                    out = _return_config_info(dev)
                  elif command[0] == 'set_configuration':
                    # Cannot set configuration twice on a same device
                    try:
                      out = getattr(dev, command[0])(*command[1:])
                    except IOError:
                      out = None
                  else:
                    # Specific syntax for device methods
                    out = getattr(dev, command[0])(*command[1:])
                else:
                  # If command is not str or list, there's an issue
                  out = TypeError("Wrong type for the command")
                  setattr(namespace, 'answer' + str(block), out)
                  setattr(namespace, 'answer' + str(block) + "'", out)
                  answer_event.set()
                  break
                if command_event.is_set():
                  # Won't send an answer if a command_event is set
                  # This is for avoiding confusion when switching from one block
                  # to the next as the command and answer events are shared by
                  # all the blocks
                  continue
                if out is None:
                  # The None commands and answers have a special meaning, so
                  # changing the Nones into ''
                  out = ''
                # Actually sending the answer
                setattr(namespace, 'answer' + str(block), out)
                setattr(namespace, 'answer' + str(block) + "'", out)
                answer_event.set()
              elif stop_event.is_set() or \
                  (all(block['left'] for key, block in blocks.items())
                   and blocks):
                # Exiting if necessary
                break
              else:
                # The timeouts are buggy so double-checking
                if command_event.wait(timeout=1):
                  continue
                # The timeouts are buggy so triple-checking
                elif time() - t < 1:
                  continue
                # In case a block doesn't respond, moving to the next one
                # Before that the command and answers are reset
                setattr(namespace, 'answer' + str(block), None)
                setattr(namespace, 'answer' + str(block) + "'", None)
                answer_event.clear()
                done_event.set()
                # And the block is put back in queue to give it another chance
                queue.put_nowait(block)
                break

            except KeyboardInterrupt:
              # Looping again in case of a CTRL+C or SIGINT
              command_event.clear()
              answer_event.clear()
              continue
        try:
          block = None
        except KeyboardInterrupt:
          block = None
    except (RuntimeError, ConnectionResetError,
            BrokenPipeError, AssertionError):
      # The USB server never raises errors itself, it just stops silently
      pass
