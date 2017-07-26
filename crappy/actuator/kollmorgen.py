from pymodbus.client.sync import ModbusTcpClient
from ..tool import convert_data


class KollMorgen(object):
  """
  Main class to test communication with kollmorgen variator.
  Every variable and its address has been defined in the Kollmorgen
  Integrated Suite. To add or remove some, update the dictionaries below.
  There is 3 motors, so the tens are for each motor, the units for the address.

  Args:
    - host: variator's IP address.
    - port: port for modbus communication (default: 502)
  """

  def __init__(self, **kwargs):
    host = kwargs.pop("host", "192.168.0.109")
    port = kwargs.pop("port", 502)
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
    """
    Toggles power of given motor. Maybe not the most intelligent way to
    handle power, though...
    """
    address = int(str(motor) + str(self.coil_addresses["power"]))
    state = self.variator.read_coils(address)
    self.variator.write_coil(address, not state.bits[0])

  def clear_errors(self):
    """
    If errors occured, it must be clear in order to continue the program.
    """
    axis_states = self.variator.read_input_registers(address=1, count=3)

    for index, err in enumerate(axis_states.registers):
      if not err == 1:
        motor = index + 1
        address = int(str(motor) + str(self.coil_addresses["ack_error"]))
        self.variator.write_coil(address, True)
        print('Cleared error (AxisState %i) in motor %i' % (err, motor))

  def set_speed(self, motor, speed):
    """
    Writes to variator desired speed (signed), and its direction. Applies to
    every motor movement.
    """
    address_hld = int(str(motor) + str(self.hldreg_addresses["velocity"]))
    self.variator.write_register(address_hld, abs(speed))
    address_hld_direction = int(str(motor) + str(self.hldreg_addresses[
                                                   "direction"]))
    if speed > 0:
      self.variator.write_register(address_hld_direction, 0)
    else:
      self.variator.write_register(address_hld_direction, 1)

  def set_accelerations(self, motor, **kwargs):
    """
    To set acceleration, deceleration (for positionning) and fast
    deceleration (boolean stop).
    """
    addresses_hld = [int(str(motor) + str(self.hldreg_addresses[value])) for
                     value in kwargs.keys()]

    for index, address in enumerate(addresses_hld):
      self.variator.write_register(address, kwargs.values()[index])

  def start_rotation(self, motor):
    """
    Sets the rotation of specifed motor at specified speed (signed).
    """
    address_coil = int(str(motor) + str(self.coil_addresses["move_vel"]))
    self.variator.write_coil(address_coil, True)

  def stop(self, motor):
    """
    Stops the motor movement.
    """
    address = int(str(motor) + str(self.coil_addresses["stop"]))
    self.variator.write_coil(address, True)

  def set_rotation(self, motor, rotation):
    """
    To set a rotation (in degrees) of the motor axis. rotation is signed.
    (not working atm: put negative ints in holding registers.)
    """
    address_coil = int(str(motor) + str(self.coil_addresses["move_rel"]))
    address_hld = int(str(motor) + str(self.hldreg_addresses["distance"]))

    # rotation = rotation * int(65535/720) + 32767
    data = convert_data.float32_to_data(rotation)
    self.variator.write_registers(address_hld, data)
    self.variator.write_coil(address_coil, True)

  def set_position(self, motor, position):
    address_coil = int(str(motor) + str(self.coil_addresses["move_abs"]))
    address_hld = int(str(motor) + str(self.hldreg_addresses["position"]))

    data = convert_data.float32_to_data(position)
    self.variator.write_registers(address_hld, data)
    self.variator.write_coil(address_coil, True)

  def read_position(self, motor):
    if not motor == "all":
      address_inpreg = int(str(motor) + str(self.inpreg_addresses[
                                              "act_position"]))
      read = self.variator.read_input_registers(address_inpreg, 2)
      converted = convert_data.data_to_float32(read.registers)
    else:
      converted = []
      for motor_adr in self.motor_addresses:
        address_inpreg = int(str(motor_adr) + str(self.inpreg_addresses[
                                                    "act_position"]))
        read = self.variator.read_input_registers(address_inpreg, 2)
        converted.append(convert_data.data_to_float32(read.registers))
    return converted

  def read_speed(self, motor):
    if not motor == "all":
      address_inpreg = int(str(motor) + str(self.inpreg_addresses[
                                              "act_speed"]))
      read = self.variator.read_input_registers(address_inpreg, 2)
      converted = convert_data.data_to_float32(read.registers)
    else:
      converted = []
      for motor_adr in self.motor_addresses:
        address_inpreg = int(str(motor_adr) + str(self.inpreg_addresses[
                                                    "act_speed"]))
        read = self.variator.read_input_registers(address_inpreg, 2)
        converted.append(convert_data.data_to_float32(read.registers))
    return converted
