# coding: utf-8

from time import time, sleep
from typing import Optional
import logging

from ..meta_inout import InOut
from ...tool.ft232h import FT232HServer as FT232H, USBArgsType

# Register Map
NAU7802_Scale_Registers = {'PU_CTRL': 0x00,
                           'CTRL1': 0x01,
                           'CTRL2': 0x02,
                           'OCAL1_B2': 0x03,
                           'OCAL1_B1': 0x04,
                           'OCAL1_B0': 0x05,
                           'GCAL1_B3': 0x06,
                           'GCAL1_B2': 0x07,
                           'GCAL1_B1': 0x08,
                           'GCAL1_B0': 0x09,
                           'OCAL2_B2': 0x0A,
                           'OCAL2_B1': 0x0B,
                           'OCAL2_B0': 0x0C,
                           'GCAL2_B3': 0x0D,
                           'GCAL2_B2': 0x0E,
                           'GCAL2_B1': 0x0F,
                           'GCAL2_B0': 0x10,
                           'I2C_CONTROL': 0x11,
                           'ADCO_B2': 0x12,
                           'ADCO_B1': 0x13,
                           'ADCO_B0': 0x14,
                           'ADC': 0x15,
                           'OTP_B1': 0x16,
                           'OTP_B0': 0x17,
                           'PGA': 0x1B,
                           'PGA_PWR': 0x1C,
                           'DEVICE_REV': 0x1F}

# Bits within the PU_CTRL register
NAU7802_PU_CTRL_Bits = {'PU_CTRL_RR': 0,
                        'PU_CTRL_PUD': 1,
                        'PU_CTRL_PUA': 2,
                        'PU_CTRL_PUR': 3,
                        'PU_CTRL_CS': 4,
                        'PU_CTRL_CR': 5,
                        'PU_CTRL_OSCS': 6,
                        'PU_CTRL_AVDDS': 7}

# Bits within the CTRL2 register
NAU7802_CTRL2_Bits = {'CTRL2_CALMOD': 0,
                      'CTRL2_CALS': 2,
                      'CTRL2_CAL_ERROR': 3,
                      'CTRL2_CRS': 4,
                      'CTRL2_CHS': 7}

# Bits within the PGA PWR register
NAU7802_PGA_PWR_Bits = {'PGA_PWR_PGA_CURR': 0,
                        'PGA_PWR_ADC_CURR': 2,
                        'PGA_PWR_MSTR_BIAS_CURR': 4,
                        'PGA_PWR_PGA_CAP_EN': 7}

# Allowed Low drop out regulator voltages
NAU7802_LDO_Values = {2.4: 0b111,
                      2.7: 0b110,
                      3.0: 0b101,
                      3.3: 0b100,
                      3.6: 0b011,
                      3.9: 0b010,
                      4.2: 0b001,
                      4.5: 0b000}

# Allowed gains
NAU7802_Gain_Values = {1: 0b000,
                       2: 0b001,
                       4: 0b010,
                       8: 0b011,
                       16: 0b100,
                       32: 0b101,
                       64: 0b110,
                       128: 0b111}

# Allowed samples per second
NAU7802_SPS_Values = {10: 0b000,
                      20: 0b001,
                      40: 0b010,
                      80: 0b011,
                      320: 0b111}

# Calibration state
NAU7802_Cal_Status = {'CAL_SUCCESS': 0,
                      'CAL_IN_PROGRESS': 1,
                      'CAL_FAILURE': 2}

NAU7802_VREF = 3.3


class NAU7802FT232H(InOut):
  """This class can read values from a NAU7802 load cell conditioner.

  It is similar to the :class:`~crappy.inout.NAU7802` class, except this class 
  is specific for use with an :class:`~crappy.tool.ft232h.FT232H` USB to I2C
  converter.

  This load cell conditioner is a low-cost 24-bits, single-channel conditioner,
  that can read up to 320 samples per second. It communicates over the I2C
  protocol. The returned value of the InOut is in Volts by default, but can be
  converted to Newtons using the ``gain`` and ``offset`` arguments.

  .. versionadded:: 2.0.0
  """

  ft232h = True

  def __init__(self,
               device_address: int = 0x2A,
               gain_hardware: int = 128,
               sample_rate: int = 80,
               int_pin: Optional[str] = None,
               gain: float = 1,
               offset: float = 0,
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments.

    Args:
      device_address: The I2C address of the NAU7802. It is impossible to
        change this address, so it is not possible to have several NAU7802
        connected on the same I2C bus.
      gain_hardware: The gain to be used by the programmable gain amplifier.
        Setting a high gain allows reading small voltages with a better
        precision, but it might saturate the sensor for higher voltages.
        Available gains are:
        ::

          1, 2, 4, 8, 16, 32, 64, 128

      sample_rate: The sample rate for data conversion. The higher the rate,
        the greater the noise. Available sample rates are:
        ::

          10, 20, 40, 80, 320

      int_pin: Optionally, reads the end of conversion signal from the polarity
        of a GPIO rather than from an I2C register. Speeds up the reading and
        decreases the traffic on the bus, but requires one extra wire. Give the
        name of the GPIO in the format `Dx` or `Cx`.
      gain: Allows to tune the output value according to the formula :
        :math:`output = gain * tension + offset`.
      offset: Allows to tune the output value according to the formula :
        :math:`output = gain * tension + offset`.
      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.
    """

    self._bus = None

    super().__init__()

    (block_index, block_lock, command_file, answer_file, shared_lock,
     current_block) = _ft232h_args

    self._bus = FT232H(mode='I2C',
                       block_index=block_index,
                       current_block=current_block,
                       command_file=command_file,
                       answer_file=answer_file,
                       block_lock=block_lock,
                       shared_lock=shared_lock)

    self._device_address = device_address

    if gain_hardware not in NAU7802_Gain_Values:
      raise ValueError("gain_hardware should be in {}".format(list(
        NAU7802_Gain_Values.keys())))
    else:
      self._gain_hardware = gain_hardware

    if sample_rate not in NAU7802_SPS_Values:
      raise ValueError("sample_rate should be in {}".format(list(
        NAU7802_SPS_Values.keys())))
    else:
      self._sample_rate = sample_rate

    if int_pin is not None and not isinstance(int_pin, str):
      raise TypeError('int_pin should be a string when using the ft232h '
                      'backend !')
    self._int_pin = int_pin

    self._gain = gain
    self._offset = offset

  def open(self) -> None:
    """Initializes the I2C communication and the device."""

    if not self._is_connected():
      raise IOError("The NAU7802 is not connected")

    self.log(logging.INFO, "Setting up the NAU7802")

    # Resetting the device
    self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_RR'],
                  NAU7802_Scale_Registers['PU_CTRL'], 1)
    sleep(0.001)
    self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_RR'],
                  NAU7802_Scale_Registers['PU_CTRL'], 0)

    # Powering up the device - takes approx 200µs
    self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_PUD'],
                  NAU7802_Scale_Registers['PU_CTRL'], 1)
    self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_PUA'],
                  NAU7802_Scale_Registers['PU_CTRL'], 1)

    t0 = time()
    while not self._get_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_PUR'],
                            NAU7802_Scale_Registers['PU_CTRL']):
      if time() - t0 > 0.1:
        raise TimeoutError("Waited too long for the device to power up !")

    # Setting the Low Drop Out voltage to 3.3V and setting the gain
    value = NAU7802_Gain_Values[self._gain_hardware]
    self._bus.write_byte_data(self._device_address,
                              NAU7802_Scale_Registers['CTRL1'], value)
    self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_AVDDS'],
                  NAU7802_Scale_Registers['PU_CTRL'], 1)

    # Setting the sample rate
    self._bus.write_byte_data(self._device_address,
                              NAU7802_Scale_Registers['CTRL2'],
                              NAU7802_SPS_Values[self._sample_rate] << 4)

    # Turning off CLK_CHP
    self._bus.write_byte_data(self._device_address,
                              NAU7802_Scale_Registers['ADC'], 0x30)

    # Enabling 330pF decoupling cap on channel 2
    self._set_bit(NAU7802_PGA_PWR_Bits['PGA_PWR_PGA_CAP_EN'],
                  NAU7802_Scale_Registers['PGA_PWR'], 1)

    # Re-calibrating the analog front-end
    self._set_bit(NAU7802_CTRL2_Bits['CTRL2_CALS'],
                  NAU7802_Scale_Registers['CTRL2'], 1)

    t0 = time()
    while self._cal_afe_status() != NAU7802_Cal_Status['CAL_SUCCESS']:
      if time() - t0 > 1:
        raise TimeoutError("Waited too long for the calibration to occur !")
      if self._cal_afe_status() == NAU7802_Cal_Status['CAL_FAILURE']:
        raise IOError("Calibration failed !")

  def get_data(self) -> list[float]:
    """Reads the registers containing the conversion result.

    The output is in Volts by default, and can be converted to Newtons using
    gain and offset.

    Returns:
      A :obj:`list` containing the timeframe and the output value.
    """

    # Waiting for data to be ready
    t0 = time()
    while not self._data_available():
      if time() - t0 > 0.5:
        raise TimeoutError('Waited too long for data to be ready !')

    out = [time()]

    # Reading the output data
    block = self._bus.read_i2c_block_data(self._device_address,
                                          NAU7802_Scale_Registers['ADCO_B2'],
                                          3)
    self.log(logging.DEBUG,
             f"Read {block} from register {NAU7802_Scale_Registers['ADCO_B2']}"
             f" at address {self._device_address}")
    value_raw = (block[0] << 16) | (block[1] << 8) | block[2]

    # Converting raw data into Volts or Newtons
    if block[0] >> 7:
      value_raw -= 2 ** 24
    value = value_raw / 2 ** 23 * 0.5 * NAU7802_VREF / self._gain_hardware
    out.append(self._offset + self._gain * value)
    return out

  def close(self) -> None:
    """Powers down the device."""

    if self._bus is not None:
      # Powering down the device
      self.log(logging.INFO, "Powering down the NAU7802")
      self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_PUD'],
                    NAU7802_Scale_Registers['PU_CTRL'], 0)
      sleep(0.001)
      self._set_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_PUA'],
                    NAU7802_Scale_Registers['PU_CTRL'], 0)
      self.log(logging.INFO, "Closing the I2C connection to the NAU7802")
      self._bus.close()

  def _is_connected(self) -> bool:
    """Tries reading a byte from the device.

    Returns:
      :obj:`True` if reading was successful, else :obj:`False`
    """

    try:
      self._bus.read_byte(self._device_address)
      return True
    except IOError:
      return False

  def _data_available(self) -> bool:
    """Returns :obj:`True` if data is available, :obj:`False` otherwise."""

    # EOC signal from the I2C communication
    if self._int_pin is None:
      return self._get_bit(NAU7802_PU_CTRL_Bits['PU_CTRL_CR'],
                           NAU7802_Scale_Registers['PU_CTRL'])
    # EOC signal from a GPIO
    else:
      return bool(self._bus.get_gpio(self._int_pin))

  def _set_bit(self,
               bit_number: int,
               register_address: int,
               bit: int) -> None:
    """Sets a given bit in the specified register.

    Args:
      bit_number: Position of the bit in the register, as an :obj:`int`.
      register_address: Index of the register, as an :obj:`int`.
      bit: Value of the bit, as an :obj:`int`.
    """

    value = self._bus.read_i2c_block_data(self._device_address,
                                          register_address, 1)[0]
    self.log(logging.DEBUG, f"Read {value} from register {register_address}"
                            f" at address {self._device_address}")
    if bit:
      value |= (1 << bit_number)
    else:
      value &= ~(1 << bit_number)
      self.log(logging.DEBUG, f"Writing {value} to register {register_address}"
                              f" at address {self._device_address}")
    self._bus.write_byte_data(self._device_address, register_address, value)

  def _get_bit(self,
               bit_number: int,
               register_address: int) -> bool:
    """Reads a given bit in the specified register.

    Args:
      bit_number: Position of the bit in the register, as an :obj:`int`.
      register_address: Index of the register, as an :obj:`int`.

    Returns:
      :obj:`True` if the bit value is 1, else :obj:`False`.
    """

    value = self._bus.read_i2c_block_data(self._device_address,
                                          register_address, 1)[0]
    self.log(logging.DEBUG, f"Read {value} from register {register_address}"
                            f" at address {self._device_address}")
    value = value >> bit_number & 1
    return bool(value)

  def _cal_afe_status(self) -> int:
    """Reads the calibration status bits.

    Returns:
      The :obj:`int` value corresponding to the current calibration status.
    """

    if self._get_bit(NAU7802_CTRL2_Bits['CTRL2_CAL_ERROR'],
                     NAU7802_Scale_Registers['CTRL2']):
      return NAU7802_Cal_Status['CAL_FAILURE']
    elif self._get_bit(NAU7802_CTRL2_Bits['CTRL2_CALS'],
                       NAU7802_Scale_Registers['CTRL2']):
      return NAU7802_Cal_Status['CAL_IN_PROGRESS']
    else:
      return NAU7802_Cal_Status['CAL_SUCCESS']
