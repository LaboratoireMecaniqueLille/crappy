# coding: utf-8

import time
from typing import List, Union
from .inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")

try:
  import spidev
except (ModuleNotFoundError, ImportError):
  spidev = OptionalModule("spidev")

# ADS1256 gain channel
Ads1256_gain = {1: 0b000,
                2: 0b001,
                4: 0b010,
                8: 0b011,
                16: 0b100,
                32: 0b101,
                64: 0b110}

# ADS1256 data rate
Ads1256_drate = {2.5: 0x03,
                 5: 0x13,
                 10: 0x20,
                 15: 0x33,
                 25: 0x43,
                 30: 0x53,
                 50: 0x63,
                 60: 0x72,
                 100: 0x82,
                 500: 0x92,
                 1000: 0xA1,
                 2000: 0xB0,
                 3750: 0xC0,
                 7500: 0xD0,
                 15000: 0xE0,
                 30000: 0xF0}

# Ads1256 registration definition
Ads1256_reg = {'REG_STATUS': 0,
               'REG_MUX': 1,
               'REG_ADCON': 2,
               'REG_DRATE': 3,
               'REG_IO': 4,
               'REG_OFC0': 5,
               'REG_OFC1': 6,
               'REG_OFC2': 7,
               'REG_FSC0': 8,
               'REG_FSC1': 9,
               'REG_FSC2': 10}

# Ads1256 command definition
Ads1256_cmd = {'CMD_WAKEUP': 0x00,
               'CMD_RDATA': 0x01,
               'CMD_RDATAC': 0x03,
               'CMD_SDATAC': 0x0F,
               'CMD_RREG': 0x10,
               'CMD_WREG': 0x50,
               'CMD_SELFCAL': 0xF0,
               'CMD_SELFOCAL': 0xF1,
               'CMD_SELFGCAL': 0xF2,
               'CMD_SYSOCAL': 0xF3,
               'CMD_SYSGCAL': 0xF4,
               'CMD_SYNC': 0xFC,
               'CMD_STANDBY': 0xFD,
               'CMD_RESET': 0xFE}

# Dac8532 channels definition
Dac8532_chan = {0: 0x10,
                1: 0x24}

# Waveshare AD/DA pins definition
AD_DA_pins = {'RST_PIN_ADS': 18,
              'CS_PIN_ADS': 22,
              'DRDY_PIN_ADS': 17,
              'CS_PIN_DAC': 23}


class Waveshare_ad_da(InOut):
  """Class for controlling Waveshare's AD/DA hat.

  The Waveshare_ad_da InOut block is meant for communicating with Waveshare's
  AD/DA Raspberry Pi hat, using the SPI protocol and the GPIOs. It allows to
  read values from the 8-channels ADC and/or to set the 2-channels DAC.

  Warning:
    This class is specifically meant to be used on a Raspberry Pi. See
    :ref:`Waveshare AD/DA FT232H` for use with FTDI's FT232H.
  """

  def __init__(self,
               dac_channels: List[str] = None,
               adc_channels: List[str] = None,
               gain_hardware: int = 1,
               v_ref: float = 3.3,
               gain: float = 1,
               offset: float = 0,
               sample_rate: Union[int, float] = 100) -> None:
    """Checks the arguments validity.

    Args:
      dac_channels (:obj:`list`, optional): A :obj:`list` of :obj:`str`
        representing the channels to be set. The syntax for each string is
        'DACi' with i being either `0` or `1`.
      adc_channels (:obj:`list`, optional): A :obj:`list` of :obj:`str`
        representing the channels to be read. The syntax for all strings is
        either:
        ::

          'ADi' (i in range(8))

        or else:
        ::

          'ADi - ADj' (i, j in range(8))

      gain_hardware (:obj:`int`, optional): The gain to be used by the
        programmable gain amplifier. Setting a high gain allows to read small
        voltages with a better precision, but it might saturate the sensor for
        higher voltages. The available gain values are:
        ::

          1, 2, 4, 8, 16, 32, 64

      v_ref (:obj:`float`, optional): The voltage reference set by the `VREF`
        jumper. When reading single inputs, ``v_ref`` is the value the ADC
        compares the signals with. In a similar way, the maximum output voltage
        of the DAC is ``v_ref``. `3.3` and `5` are the only possible values for
        this setting, as the Raspberry Pi can only provide `3.3V` and `5V`.
      gain (:obj:`float`, optional): Allows to tune the output values of the
        DAC according to the formula:
        ::

          output = gain * tension + offset.

        The same gain applies to all of the outputs.
      offset (:obj:`float`, optional): Allows to tune the output values of the
        ADC according to the formula:
        ::

          output = gain * tension + offset.

        The same offset applies to all of the outputs.
      sample_rate (optional): The ADC data output rate in SPS. The available
        values are:
        ::

          2.5, 5, 10, 15, 25, 30, 50, 60, 100, 500,
          1000, 2000, 3750, 7500, 15000, 30000

    Warning:
      - ``adc_channels``:
        For reading single inputs the `JMP_AGND` jumper should normally be
        connected, whereas it should be disconnected for reading differential
        inputs. It is however possible to set a different reference than `AGND`
        for single input measurements, in which case the `JMP_AGND` jumper
        should not be connected and the voltage reference should be plugged
        in the `AINCOM` pin.

        The AD/DA offers the possibility to read single inputs or
        differential inputs, but not both at the same time ! This is due to
        the `JMP_AGND` jumper.
        For measuring both input types simultaneously, is it necessary to
        connect `AGND` to one of the channels (for example `AD0`). Then all
        single inputs `'ADi'` should be replaced by `'ADi - AD0'`. They are
        then considered as differential inputs.

        The ADC channels voltages should not be lower than `AGND-0.1V`, and not
        be greater than `AGND+5.1V`. This is independent from `VREF` value.

    Note:
      - ``dac_channels``:
        As there are 2 DAC channels on the AD/DA, only `1` or `2` strings can
        be given for the dac_channels argument.

      - ``adc_channels``:
        If multiple channels to read are given, they are read in a sequential
        way. This means that there's a small delay between each acquisition,
        and the timeframe is thus less accurate for the last channels than
        for the first ones. If time precision matters it is preferable to
        read as few channels as possible !

      - ``vref``:
        `VREF` can be set independently from the chosen `VCC` value. The `VCC`
        value has no influence on the ADC behaviour as it is always powered
        up with `5V`. Same goes for the DAC.
    """

    InOut.__init__(self)

    if gain_hardware not in Ads1256_gain:
      raise ValueError("gain_hardware should be in {}".format(list(
        Ads1256_gain.keys())))
    else:
      self._gain_hardware = gain_hardware

    if sample_rate not in Ads1256_drate:
      raise ValueError("sample_rate should be in {}".format(list(
        Ads1256_drate.keys())))
    else:
      self._sample_rate = sample_rate

    if v_ref not in [3.3, 5]:
      raise ValueError("v_ref should be either 3.3 or 5")
    else:
      self._v_ref = v_ref

    self._channel_set = False

    if dac_channels is None and adc_channels is None:
      print("Warning ! The AD/DA doesn't read nor write anything.")

    if dac_channels is not None:
      if len(dac_channels) > 2:
        raise ValueError("dac_channels length should not exceed 2")
      for chan in dac_channels:
        if chan not in ["DAC0", "DAC1"]:
          raise ValueError("Valid format for dac_channels values is 'DACi' "
                           "with i either 0 or 1")
      self._dac_channels = dac_channels
    else:
      self._dac_channels = []

    if adc_channels is not None:
      not_valid_message = "Valid formats for adc_channels values are " \
                          "either 'ADi' (i in range(8)) or 'ADi - ADj'"
      for chan in adc_channels:
        if len(chan) not in [3, 9]:
          raise ValueError(not_valid_message)
        if not chan.startswith("AD"):
          raise ValueError(not_valid_message)
        elif len(chan) == 9:
          if not chan.startswith("-", 4):
            raise ValueError(not_valid_message)
          if not chan.startswith("AD", 6):
            raise ValueError(not_valid_message)
      if not (all(len(chan) == 3 for chan in adc_channels) or
              all(len(chan) == 9 for chan in adc_channels)):
        raise ValueError("It is not possible to have both single and "
                         "differential inputs, see documentation for more "
                         "info")
      self._adc_channels = adc_channels
    else:
      self._adc_channels = []

    self._gain = gain
    self._offset = offset

    self._SPI = spidev.SpiDev(0, 0)
    self._channels_read, self._channels_write = [], []

  def open(self) -> None:
    """Sets the SPI communication, the GPIOs and the device."""

    # Setting the GPIOs for communicating with the AD/DA
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(AD_DA_pins['RST_PIN_ADS'], GPIO.OUT)
    GPIO.setup(AD_DA_pins['CS_PIN_ADS'], GPIO.OUT)
    GPIO.setup(AD_DA_pins['DRDY_PIN_ADS'], GPIO.IN,
               pull_up_down=GPIO.PUD_UP)
    GPIO.setup(AD_DA_pins['CS_PIN_DAC'], GPIO.OUT)

    # Setting the SPI
    self._SPI.max_speed_hz = 40000
    self._SPI.mode = 1
    self._SPI.no_cs = True
    self._reset()

    # Setting the ADS according to the user parameters
    buf = [0x00, 0x00]
    buf[0] |= Ads1256_gain[self._gain_hardware]
    buf[1] |= Ads1256_drate[self._sample_rate]
    GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.LOW)
    self._SPI.writebytes([Ads1256_cmd['CMD_WREG'] |
                         Ads1256_reg['REG_ADCON'], 0x01] + buf)
    GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.HIGH)
    time.sleep(0.001)

    # Setting the lists of channels to read and write
    for chan in self._adc_channels:
      if len(chan) == 3:
        self._channels_read.append([int(chan[2]), 8])
      else:
        self._channels_read.append([int(chan[2]), int(chan[8])])

    for chan in self._dac_channels:
      self._channels_write.append(int(chan[3]))

  def get_data(self) -> list:
    """Reads data from all the user-specified ADC channels, in a sequential
    way.

    Data is returned in Volts, but this can be tuned using gain and offset.

    Returns:
      :obj:`list`: A list containing the timeframe, and then the values for
      each channel to read
    """

    out = [time.time()]

    # The values are read one channel after the other, not simultaneously
    for chan in self._channels_read:
      GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.LOW)

      # Switching channel only if necessary, except for the first loop
      if len(self._channels_read) > 1 or not self._channel_set:
        self._SPI.writebytes([Ads1256_cmd['CMD_WREG'] |
                             Ads1256_reg['REG_MUX'], 0x00,
                             (chan[0] << 4) | chan[1]])

        # The ADS has to be synchronized again when switching channel
        self._SPI.writebytes([Ads1256_cmd['CMD_SYNC']])
        self._SPI.writebytes([Ads1256_cmd['CMD_WAKEUP']])

        self._channel_set = True

      # Reading the output value
      self._wait_drdy()
      self._SPI.writebytes([Ads1256_cmd['CMD_RDATA']])
      buf = self._SPI.readbytes(3)
      GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.HIGH)

      # Converting the raw output into Volts
      out_raw = (buf[0] << 16) | (buf[1] << 8) | buf[2]
      if out_raw >> 23:
        value = self._v_ref / self._gain_hardware * (-(2 ** 23 - 1) + (
                out_raw & 0x7FFFFF)) / (2 ** 23 - 1)
      else:
        value = self._v_ref / self._gain_hardware * out_raw / (
                2 ** 23 - 1)
      out.append(self._offset + self._gain * value)

    return out

  def set_cmd(self, *cmd: float) -> None:
    """Sets the user-specified DAC channels according to the input values.

    Args:
      cmd (:obj:`float`): The input values, in Volts
    """

    # The values are set one channel after the other, not simultaneously
    for val, channel in zip(cmd, self._channels_write):
      if not 0 <= val <= self._v_ref:
        raise ValueError("Desired output voltage should be between 0 and "
                         "v_ref")
      digit = int((2 ** 16 - 1) * val / self._v_ref)
      GPIO.output(AD_DA_pins['CS_PIN_DAC'], GPIO.LOW)
      self._SPI.writebytes([Dac8532_chan[channel], digit >> 8,
                           digit & 0xFF])
      GPIO.output(AD_DA_pins['CS_PIN_DAC'], GPIO.HIGH)

  def close(self) -> None:
    """Releases the GPIOs."""

    self._SPI.close()
    GPIO.cleanup()

  @staticmethod
  def _reset() -> None:
    """Resets the ADC."""

    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.LOW)
    time.sleep(0.2)
    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.HIGH)

  @staticmethod
  def _wait_drdy() -> None:
    """Waits until the `DRDY` pin is driven low, meaning that an ADC conversion
    is ready."""

    t_wait = 0
    while GPIO.input(AD_DA_pins['DRDY_PIN_ADS']):
      time.sleep(0.0001)
      t_wait += 0.0001
      if t_wait > 1:
        raise TimeoutError("Couldn't get conversion result from the ADC")
