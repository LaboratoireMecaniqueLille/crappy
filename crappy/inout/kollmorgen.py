# coding: utf-8

from time import time
from struct import pack, unpack
from typing import List

from .inout import InOut
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


class Koll(InOut):
  """This class can communicate with a KollMorgen AKD PDMM programmable
   multi-axis controller.

   It can either drive it in speed or in position. Multiple axes can be driven.
   The values of the current speeds or positions can also be retrieved.
   """

  def __init__(self,
               axes: List[int],
               mode: str = 'position',
               host: str = '192.168.0.109',
               port: int = 502) -> None:
    """Sets the args and initializes the parent class.

    Args:
      axes: A :obj:`list` containing the motors/axes to drive, given as
        :obj:`int`.
      mode: Should be either `'speed'` or `'position'`. Whether the axes
        should be driven in speed or in position.
      host: The IP address of the variator, given as a :obj:`str`.
      port: The network port over which to communicate with the variator, as
        an :obj:`int`.
    """

    super().__init__()

    # Making sure the given mode is correct
    if mode not in ('speed', 'position'):
      raise ValueError("[KollMorgen] the mode argument should be either "
                       "'speed' or 'position' !")

    self._axes = axes
    self._mode = mode

    self._variator = ModbusTcpClient(host=host, port=port)

  def open(self) -> None:
    """Connects to the variator over modbus."""

    self._variator.connect()

  def get_data(self) -> List[float]:
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

    self._variator.close()
