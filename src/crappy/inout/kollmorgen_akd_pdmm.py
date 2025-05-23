# coding: utf-8

from time import time
from struct import pack, unpack
from typing import Literal
from collections.abc import Iterable
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule
try:
  from pymodbus.client.tcp import ModbusTcpClient
except (ModuleNotFoundError, ImportError):
  ModbusTcpClient = OptionalModule("pymodbus", "Cannot use KollMorgenVariator")


coil_addr = {'power': 0,
             'move_abs': 1,
             'move_rel': 2,
             'move_vel': 3,
             'stop': 4,
             'ack_error': 5}

reg_addr = {'position': 0,
            'distance': 2,
            'velocity': 4,
            'acc': 5,
            'dec': 6,
            'fstdec': 7,
            'direction': 8}

input_reg_addr = {'act_speed': 0,
                  'act_position': 2,
                  'axis_state': 4}


class KollmorgenAKDPDMM(InOut):
  """This class can communicate with a KollMorgen AKD PDMM programmable
   multi-axis controller.

   It can either drive it in speed or in position. Multiple axes can be driven.
   The values of the current speeds or positions can also be retrieved.
   
   .. versionadded:: 1.4.0
   .. versionchanged:: 2.0.0 renamed from *Koll* to *KollmorgenAKDPDMM*
   """

  def __init__(self,
               axes: Iterable[int],
               mode: Literal['position', 'speed'] = 'position',
               host: str = '192.168.0.109',
               port: int = 502) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      axes: An iterable (like a :obj:`list` or a :obj:`tuple`) containing the
        motors/axes to drive, given as :obj:`int`.

        .. versionchanged:: 1.5.10 renamed from *axis* to *axes*
      mode: The driving mode, should be either `'speed'` or `'position'`.

        .. versionchanged:: 1.5.10 renamed from *data* to *mode*
      host: The IP address of the variator, given as a :obj:`str`.
      port: The network port over which to communicate with the variator, as
        an :obj:`int`.
    
    .. versionremoved:: 1.5.10 *speed*, *acc*, *decc* and *labels* arguments
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._variator = None

    super().__init__()

    # Making sure the given mode is correct
    if mode not in ('speed', 'position'):
      raise ValueError("The mode argument should be either 'speed' or "
                       "'position' !")

    self._axes = axes
    self._mode = mode
    self._host = host
    self._port = port

  def open(self) -> None:
    """Connects to the variator over modbus."""

    self.log(logging.INFO, f"Initializing the TCP connection to address "
                           f"{self._host} on port {self._port}")
    self._variator = ModbusTcpClient(host=self._host, port=self._port)
    self.log(logging.INFO, f"Opening the TCP connecting to the address "
                           f"{self._host} on port {self._port}")
    self._variator.connect()

  def get_data(self) -> list[float]:
    """For each motor, reads its current speed or position depending on the
    selected mode.

    The positions or speeds are returned in the same order as the motors were
    given in the ``axes`` argument.
    """

    ret = [time()]

    for axis in self._axes:

      # Selecting the register to read depending on the selected mode
      if self._mode == 'speed':
        reg = 10 * axis + input_reg_addr["act_speed"]
      else:
        reg = 10 * axis + input_reg_addr["act_position"]

      # Actually reading the data and converting it
      read = self._variator.read_input_registers(address=reg, count=2)
      converted, *_ = unpack("=f", pack("=HH", *read.registers[::-1]))

      ret.append(converted)

    return ret

  def set_cmd(self, *cmd: float) -> None:
    """Sets either the speed or the position of all motors depending on the
    selected mode.

    If more commands than motors are given, the extra commands are ignored. If
    there are more motors than commands, only part of the motors will be set.
    
    .. versionadded:: 1.5.10
    """

    for axis, val in zip(self._axes, cmd):

      # Setting the speed of each motor
      if self._mode == 'speed':
        reg = 10 * axis + reg_addr['velocity']
        self._variator.write_register(reg, abs(val))

        direction = 10 * axis + reg_addr['direction']
        self._variator.write_register(direction, 0 if val > 0 else 1)

      # Setting the target position of each motor
      else:
        coil = 10 * axis + coil_addr['move_abs']
        reg = 10 * axis + reg_addr['position']
        data = unpack("=HH", pack("=f", val))[::-1]

        self._variator.write_registers(reg, list(data))
        self._variator.write_coil(coil, True)

  def close(self) -> None:
    """Closes the modbus connection to the variator."""

    if self._variator is not None:
      self.log(logging.INFO, f"Closing the TCP connecting to the address "
                             f"{self._host} on port {self._port}")
      self._variator.close()
