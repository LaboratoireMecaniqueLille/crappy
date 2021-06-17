from time import time
import struct

from .inout import InOut
from .._global import OptionalModule
try:
  from pymodbus.client.sync import ModbusTcpClient
except (ModuleNotFoundError, ImportError):
  ModbusTcpClient = OptionalModule("pymodbus", "Cannot use KollMorgenVariator")


def float32_to_data(num):
  return struct.unpack("=HH", struct.pack("=f", num))[::-1]


def data_to_float32(data):
  assert len(data) == 2, "float32 expects 2 registers"
  return struct.unpack("=f", struct.pack("=HH", *data[::-1]))[0]


class KollMorgenVariator:
  """Main class to test communication with kollmorgen variator.

  Every variable and its address has been defined in the Kollmorgen Integrated
  Suite.

  To add or remove some, update the dictionaries of :meth:`__init__`.

  There are 3 motors, so the tens are for each motor, the units for the
  address.
  """

  def __init__(self,
               host='192.168.0.109',
               port=502):
    """Sets the variator and defines the bits.

    Args:
      host (:obj:`str`, optional): Variator's IP address.
      port (:obj:`int`, optional): Port for modbus communication
    """

    self.variator = ModbusTcpClient(host=host, port=port)
    assert self.variator.connect(), "ERROR: could not connect to variator."

    self.motor_addresses = [1, 2, 3, 4]

    # R/W bits: coils
    self.coil_addresses = {
      'power': 0,
      'move_abs': 1,
      'move_rel': 2,
      'move_vel': 3,
      'stop': 4,
      'ack_error': 5}

    # R/W int32: holding registers
    self.hldreg_addresses = {
      'position': 0,
      'distance': 2,
      'velocity': 4,
      'acc': 5,
      'dec': 6,
      'fstdec': 7,
      'direction': 8}

    # R bits: input bits, not used.
    # R int32: input registers.

    self.inpreg_addresses = {
      'act_speed': 0,
      'act_position': 2,
      'axis_state': 4
    }

  def toggle_power(self, motor):
    """Toggles power of given motor. Maybe not the most intelligent way to
    handle power, though..."""

    address = int(str(motor) + str(self.coil_addresses["power"]))
    state = self.variator.read_coils(address)
    self.variator.write_coil(address, not state.bits[0])

  def clear_errors(self):
    """If errors occurred, it must be clear in order to continue the program.
    """

    axis_states = self.variator.read_input_registers(address=1, count=3)

    for index, err in enumerate(axis_states.registers):
      if not err == 1:
        motor = index + 1
        address = int(str(motor) + str(self.coil_addresses["ack_error"]))
        self.variator.write_coil(address, True)
        print('Cleared error (AxisState %i) in motor %i' % (err, motor))

  def set_speed(self, motor, speed):
    """Writes to variator desired speed (signed), and its direction. Applies to
    every motor movement (rotations, positioning...)."""

    address_hld = int(str(motor) + str(self.hldreg_addresses["velocity"]))
    self.variator.write_register(address_hld, abs(speed))
    address_hld_direction = int(str(motor) + str(self.hldreg_addresses[
      "direction"]))
    if speed > 0:
      self.variator.write_register(address_hld_direction, 0)
    else:
      self.variator.write_register(address_hld_direction, 1)

  def set_accelerations(self, motor, **kwargs):
    """To set acceleration, deceleration (for positioning) and fast
    deceleration (boolean stop)."""

    for address, value in kwargs.items():
      self.variator.write_register(int(str(motor) +
                                       str(self.hldreg_addresses[address])),
                                   value)

  def start_rotation(self, motor):
    """Sets the rotation of specified motor at specified speed (signed)."""

    address_coil = int(str(motor) + str(self.coil_addresses["move_vel"]))
    self.variator.write_coil(address_coil, True)

  def stop(self, motor):
    """Stops the motor movement."""

    address = int(str(motor) + str(self.coil_addresses["stop"]))
    self.variator.write_coil(address, True)

  def set_rotation(self, motor, rotation):
    """To set a rotation (in degrees) of the motor axis. Rotation is signed."""

    address_coil = int(str(motor) + str(self.coil_addresses["move_rel"]))
    address_hld = int(str(motor) + str(self.hldreg_addresses["distance"]))

    data = float32_to_data(rotation)
    self.variator.write_registers(address_hld, data)
    self.variator.write_coil(address_coil, True)

  def set_position(self, motor, position):
    """To set a position (in degrees), absolute value."""

    address_coil = int(str(motor) + str(self.coil_addresses["move_abs"]))
    address_hld = int(str(motor) + str(self.hldreg_addresses["position"]))

    data = float32_to_data(position)
    self.variator.write_registers(address_hld, data)
    self.variator.write_coil(address_coil, True)

  def read_position(self, motor):
    """To read position of motor. Returns a :obj:`float`."""

    if not motor == "all":
      # If 1 axis is needed
      address_inpreg = int(str(motor) + str(self.inpreg_addresses[
        "act_position"]))
      read = self.variator.read_input_registers(address_inpreg, 2)
      converted = data_to_float32(read.registers)
    else:
      converted = []
      # Reads 40 first addresses, and then extracts values from the length 40
      #  list. Much more efficient, limits communication time.
      read = self.variator.read_input_registers(0, 44)
      for motor_adr in self.motor_addresses:
        address_inpreg = int(str(motor_adr) + str(self.inpreg_addresses[
         "act_position"]))

        data = read.registers[address_inpreg:address_inpreg + 2]
        converted.append(data_to_float32(data))
    return converted

  def read_speed(self, motor):
    """Reads speed of each motor."""

    if not motor == "all":
      address_inpreg = int(str(motor) + str(self.inpreg_addresses[
          "act_speed"]))
      read = self.variator.read_input_registers(address_inpreg, 2)
      converted = data_to_float32(read.registers)

    else:
      converted = []
      read = self.variator.read_input_registers(0, 44)
      for motor_adr in self.motor_addresses:
        address_inpreg = int(str(motor_adr) + str(self.inpreg_addresses[
          "act_speed"]))

        data = read.registers[address_inpreg:address_inpreg + 2]
        converted.append(data_to_float32(data))
    return converted


class Koll(InOut):
  """Class to communicate to Kollmorgen devices via Crappy."""

  def __init__(self,
               data='position',
               axis='all',
               speed=360,
               acc=3600,
               decc=3600,
               labels=None,
               host='192.168.0.109',
               port=502):
    InOut.__init__(self)

    self.data = data
    self.axis = axis
    self.speed = speed
    self.acc = acc
    self.decc = decc

    if self.axis == "all":
      default_label = ["t(s)"] + [str(i) for i in range(1, 4)]
    else:
      default_label = ["t(s)", str(self.axis)]

      # NB: I have trouble defining default args for labels. It seems that if
      #  kwargs doesn't contain "labels", I cannot put another default value
      # than ("t(s)", "1") as defined in InOut parent class...

    self.labels = default_label if labels is None else labels
    self.variator = KollMorgenVariator(host=host, port=port)

  def open(self):
    pass

  def get_data(self):
    if self.data == "speed":
      if not self.axis == "all":
        ret = [time(), self.variator.read_speed(self.axis)]
      else:
        ret = [time()] + self.variator.read_speed(self.axis)

    elif self.data == "position":
      if not self.axis == "all":
        ret = [time(), self.variator.read_position(self.axis)]
      else:
        ret = [time()] + self.variator.read_position(self.axis)
    else:
      return
    return ret

  def close(self):
    pass
