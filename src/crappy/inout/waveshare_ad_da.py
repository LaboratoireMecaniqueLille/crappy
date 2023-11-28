# coding: utf-8

import time
from time import sleep, time
from re import fullmatch, findall
from typing import Union, Optional, List, Iterable
import logging

from .meta_inout import InOut
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


class WaveshareADDA(InOut):
  """Class for controlling Waveshare's AD/DA Raspberry Pi hat.

  It communicates over the SPI protocol and the GPIOs. It allows to
  read values from the 8-channels ADC and/or to set the 2-channels DAC. The hat
  can acquire up to 30000 samples per second, although this data rate is
  impossible to achieve using Crappy.

  Important:
    This class is specifically meant to be used on a Raspberry Pi.
  """

  def __init__(self,
               dac_channels: Optional[Iterable[str]] = None,
               adc_channels: Optional[Iterable[str]] = None,
               gain_hardware: int = 1,
               v_ref: float = 3.3,
               gain: float = 1,
               offset: float = 0,
               sample_rate: Union[int, float] = 100) -> None:
    """Checks the validity of the arguments.

    Args:
      dac_channels: An iterable (like a :obj:`list` or a :obj:`tuple`) of
        :obj:`str` representing the channels to be set. The syntax for each
        string is 'DACi' with i being either `0` or `1`.
      adc_channels: An iterable (like a :obj:`list` or a :obj:`tuple`) of
        :obj:`str` representing the channels to read. The syntax for all the
        strings is either:
        ::

          'ADi' (i in range(8))

        or else:
        ::

          'ADi - ADj' (i, j in range(8))

      gain_hardware: The gain to be used by the programmable gain amplifier.
        Setting a high gain allows to read small voltages with a better
        precision, but it might saturate the sensor for higher voltages. The
        available gain values are:
        ::

          1, 2, 4, 8, 16, 32, 64

      v_ref: The voltage reference set by the `VREF` jumper. When reading
        single inputs, ``v_ref`` is the value the ADC compares the signals
        with. In a similar way, the maximum output voltage of the DAC is
        ``v_ref``. `3.3` and `5` are the only possible values for this setting,
        as the Raspberry Pi can only provide `3.3V` and `5V`.
      gain: Allows to tune the output values of the ADC according to the
        formula : :math:`output = gain * tension + offset`. The same gain
        applies to all the outputs.
      offset: Allows to tune the output values of the ADC according to the
        formula : :math:`output = gain * tension + offset`. The same offset
        applies to all the outputs.
      sample_rate: The ADC data output rate in SPS. The available values are:
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

        The AD/DA offers the possibility to read single inputs or differential
        inputs, but not both at the same time ! This is due to the `JMP_AGND`
        jumper. For measuring both input types simultaneously, is it necessary
        to connect `AGND` to one of the channels (for example `AD0`). Then all
        single inputs `'ADi'` should be replaced by `'ADi - AD0'`. They are
        then considered as differential inputs.

        The ADC channels voltages should not be lower than `AGND-0.1V`, and not
        be greater than `AGND+5.1V`. This is independent of `VREF` value.

    Note:
      - ``adc_channels``:
        If multiple channels to read are given, they are read in a sequential
        way. This means that there's a small delay between each acquisition,
        and the timeframe is thus less accurate for the last channels than
        for the first ones. If time precision matters it is preferable to
        read as few channels as possible !

      - ``vref``:
        `VREF` can be set independently of the chosen `VCC` value. The `VCC`
        value has no influence on the ADC behaviour as it is always powered
        up with `5V`. Same goes for the DAC.
    """

    self._bus = None

    super().__init__()

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
      self.log(logging.WARNING, "The AD/DA doesn't read nor write anything")

    self._channels_write = []
    if dac_channels is not None:
      for chan in dac_channels:
        # Checking if the format matches
        if fullmatch(r'DAC[0-1]', chan) is None:
          raise ValueError("Valid format for dac_channels values is 'DACi' "
                           "with i either 0 or 1")
        else:
          self._channels_write.append(int(findall(r'\d', chan)[0]))

      self.log(logging.DEBUG, f"Channels to write to: {self._channels_write}")

    self._channels_read = []
    if adc_channels is not None:
      for chan in adc_channels:
        # Checking if the format matches
        if fullmatch(r'AD[0-7]', chan) is None and \
              fullmatch(r'AD[0-7]\s?-\s?AD[0-7]', chan) is None:
          raise ValueError("Valid formats for adc_channels values are "
                           "either 'ADi' (i in range(8)) or 'ADi - ADj'")
        else:
          # Extracting the channel numbers
          chan = [int(i) for i in findall(r'\d', chan)]
          # If only one channel provided, it has to be compared to GND
          if len(chan) == 1:
            chan.append(8)
          # Making sure the two channels are not identical
          if chan[0] == chan[1]:
            raise ValueError("The two channels are the same !")
          self._channels_read.append(chan)

      self.log(logging.DEBUG, f"Channels to read from: {self._channels_read}")

    self._gain = gain
    self._offset = offset

    self.log(logging.INFO, "Opening the SPI communication with the AD/DA")
    self._bus = spidev.SpiDev(0, 0)

  def open(self) -> None:
    """Sets the SPI communication, the GPIOs and the device."""

    # Setting the GPIOs for communicating with the AD/DA
    self.log(logging.INFO, "Setting up the GPIOs")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(AD_DA_pins['RST_PIN_ADS'], GPIO.OUT)
    GPIO.setup(AD_DA_pins['CS_PIN_ADS'], GPIO.OUT)
    GPIO.setup(AD_DA_pins['DRDY_PIN_ADS'], GPIO.IN,
               pull_up_down=GPIO.PUD_UP)
    GPIO.setup(AD_DA_pins['CS_PIN_DAC'], GPIO.OUT)

    # Setting the SPI
    self.log(logging.INFO, "Setting up the SPI connection")
    self._bus.max_speed_hz = 40000
    self._bus.mode = 1
    self._bus.no_cs = True
    self._reset()

    # Setting the ADS according to the user parameters
    buf = [Ads1256_gain[self._gain_hardware], Ads1256_drate[self._sample_rate]]
    GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.LOW)
    cmd = [Ads1256_cmd['CMD_WREG'] | Ads1256_reg['REG_ADCON'], 0x01] + buf
    self.log(logging.DEBUG, f"Writing {cmd} to the SPI bus")
    self._bus.writebytes(cmd)
    GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.HIGH)
    sleep(0.001)

  def get_data(self) -> List[float]:
    """Reads data from all the user-specified ADC channels, in a sequential
    way.

    Data is returned in Volts, but this can be tuned using gain and offset.

    Returns:
      A :obj:`list` containing the timestamp, and then the values for each
      channel to read.
    """

    out = [time()]

    # The values are read one channel after the other, not simultaneously
    for chan in self._channels_read:
      GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.LOW)

      # Switching channel only if necessary, except for the first loop
      if len(self._channels_read) > 1 or not self._channel_set:
        cmd = [Ads1256_cmd['CMD_WREG'] | Ads1256_reg['REG_MUX'], 0x00,
               (chan[0] << 4) | chan[1]]
        self.log(logging.DEBUG, f"Writing {cmd} to the SPI bus")
        self._bus.writebytes(cmd)

        # The ADS has to be synchronized again when switching channel
        self.log(logging.DEBUG, f"Writing {[Ads1256_cmd['CMD_SYNC']]} to the "
                                f"SPI bus")
        self._bus.writebytes([Ads1256_cmd['CMD_SYNC']])
        self.log(logging.DEBUG, f"Writing {[Ads1256_cmd['CMD_WAKEUP']]} to "
                                f"the SPI bus")
        self._bus.writebytes([Ads1256_cmd['CMD_WAKEUP']])

        self._channel_set = True

      # Reading the output value
      self._wait_drdy()
      self.log(logging.DEBUG, f"Writing {[Ads1256_cmd['CMD_RDATA']]} to the "
                              f"SPI bus")
      self._bus.writebytes([Ads1256_cmd['CMD_RDATA']])
      buf = self._bus.readbytes(3)
      self.log(logging.DEBUG, f"Read {buf} from the SPI bus")
      GPIO.output(AD_DA_pins['CS_PIN_ADS'], GPIO.HIGH)

      # Converting the raw output into Volts
      out_raw = (buf[0] << 16) | (buf[1] << 8) | buf[2]
      if out_raw & 0x800000:
        out_raw -= 2 ** 24
      value = self._v_ref / self._gain_hardware * out_raw / 2 ** 23
      out.append(self._offset + self._gain * value)

    return out

  def set_cmd(self, *cmd: float) -> None:
    """Sets the user-specified DAC channels according to the input values.

    Args:
      cmd: The input values as :obj:`float`, in Volts.
    """

    # The values are set one channel after the other, not simultaneously
    for val, channel in zip(cmd, self._channels_write):
      if not 0 <= val <= self._v_ref:
        raise ValueError("Desired output voltage should be between 0 and "
                         "v_ref")
      digit = int((2 ** 16 - 1) * val / self._v_ref)
      GPIO.output(AD_DA_pins['CS_PIN_DAC'], GPIO.LOW)
      cmd = [Dac8532_chan[channel], digit >> 8, digit & 0xFF]
      self.log(logging.DEBUG, f"Writing {cmd} to the SPI bus")
      self._bus.writebytes(cmd)
      GPIO.output(AD_DA_pins['CS_PIN_DAC'], GPIO.HIGH)

  def close(self) -> None:
    """Releases the GPIOs."""

    if self._bus is not None:
      self.log(logging.INFO, "Closing the SPI communication with the AD/DA")
      self._bus.close()
    self.log(logging.INFO, "Cleaning up the GPIOs")
    GPIO.cleanup()

  @staticmethod
  def _reset() -> None:
    """Resets the ADC."""

    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.HIGH)
    sleep(0.2)
    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.LOW)
    sleep(0.2)
    GPIO.output(AD_DA_pins['RST_PIN_ADS'], GPIO.HIGH)

  @staticmethod
  def _wait_drdy() -> None:
    """Waits until the `DRDY` pin is driven low, meaning that an ADC conversion
    is ready."""

    t0 = time()
    while GPIO.input(AD_DA_pins['DRDY_PIN_ADS']):
      if time() - t0 > 1:
        raise TimeoutError("Couldn't get conversion result from the ADC")
