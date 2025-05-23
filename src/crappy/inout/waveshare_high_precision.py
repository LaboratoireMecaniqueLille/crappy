# coding : utf-8

from time import time, sleep
from re import fullmatch, findall
from typing import Union, Optional
from collections.abc import Iterable
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")

try:
  from spidev import SpiDev
except (ModuleNotFoundError, ImportError):
  SpiDev = OptionalModule("spidev")

# gain
ADS1263_GAIN = {1: 0,
                2: 1,
                4: 2,
                8: 3,
                16: 4,
                32: 5,
                64: 6}

# data rate
ADS1263_DRATE = {2.5: 0x00,
                 5: 0x01,
                 10: 0x02,
                 16.6: 0x03,
                 20: 0x04,
                 50: 0x05,
                 60: 0x06,
                 100: 0x07,
                 400: 0x08,
                 1200: 0x09,
                 2400: 0x0A,
                 4800: 0x0B,
                 7200: 0x0C,
                 14400: 0x0D,
                 19200: 0x0E,
                 38400: 0x0F}

ADS1263_CMD = {'CMD_RESET': 0x06,
               'CMD_START1': 0x08,
               'CMD_STOP1': 0x0A,
               'CMD_START2': 0x0C,
               'CMD_STOP2': 0x0E,
               'CMD_RDATA1': 0x12,
               'CMD_RDATA2': 0x14,
               'CMD_SYOCAL1': 0x16,
               'CMD_SYGCAL1': 0x17,
               'CMD_SFOCAL1': 0x19,
               'CMD_SYOCAL2': 0x1B,
               'CMD_SYGCAL2': 0x1C,
               'CMD_SFOCAL2': 0x1E,
               'CMD_RREG': 0x20,
               'CMD_RREG2': 0x00,
               'CMD_WREG': 0x40,
               'CMD_WREG2': 0x00}

# registration definition
ADS1263_REG = {'REG_ID': 0x00,
               'REG_POWER': 0x01,
               'REG_INTERFACE': 0x02,
               'REG_MODE0': 0x03,
               'REG_MODE1': 0x04,
               'REG_MODE2': 0x05,
               'REG_INPMUX': 0x06,
               'REG_OFCAL0': 0x07,
               'REG_OFCAL1': 0x08,
               'REG_OFCAL2': 0x09,
               'REG_FSCAL0': 0x0A,
               'REG_FSCAL1': 0x0B,
               'REG_FSCAL2': 0x0C,
               'REG_IDACMUX': 0x0D,
               'REG_IDACMAG': 0x0E,
               'REG_REFMUX': 0x0F,
               'REG_TDACP': 0x10,
               'REG_TDACN': 0x11,
               'REG_GPIOCON': 0x12,
               'REG_GPIODIR': 0x13,
               'REG_GPIODAT': 0x14,
               'REG_ADC2CFG': 0x15,
               'REG_ADC2MUX': 0x16,
               'REG_ADC2OFC0': 0x17,
               'REG_ADC2OFC1': 0x18,
               'REG_ADC2FSC0': 0x19,
               'REG_ADC2FSC1': 0x1A}

RST_PIN = 18
CS_PIN = 22
DRDY_PIN = 17

VREF = 5


class WaveshareHighPrecision(InOut):
  """This InOut allows acquiring data from Waveshare's High Precision HAT.

  This board features an ADS1263 32-bits ADC, which is what this InOut actually
  drives. The main specificities compared with driving just an ADS1263 are that
  the reference voltage is the board's 5V supply, and that the differential
  acquisition is improved when using pairs of channels (0 & 1, 2 & 3, etc.).

  The sample rate, hardware gain and digital filter can be adjusted. It is also
  possible to acquire data sequentially from several channels.

  The Waveshare HAT is originally meant to be used with a Raspberry Pi, but it
  can be used with any device supporting SPI as long as the wiring is correct
  and the 3.3 and 5V power are supplied.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Waveshare_high_precision* to *WaveshareHighPrecision*
  """

  def __init__(self,
               spi_port: int = 0,
               gain_hardware: int = 16,
               sample_rate: Union[int, float] = 50,
               channels: Optional[Iterable[str]] = None,
               digital_filter: int = 4,
               gain: float = 1,
               offset: float = 0) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      spi_port: The SPI port for communicating with the Waveshare HAT.
      gain_hardware: A programmable gain for the signal. Should be one of :
        ::

          1, 2, 4, 8, 16, 32

        Allows increasing the resolution of the ADC, but also divides the full
        scale range and greatly reduces the noise performance when set greater
        than 8. This value applies to all the channels.
      sample_rate: The number of samples per second to acquire. Should be one
        of:
        ::

          2.5, 5, 10, 16, 20, 50, 60, 100, 400, 1200, 2400, 4800, 7200, \
          14400, 19200, 38400

        The actual achieved sample rate might be lower depending on the
        capability of the PC and the load on the processor. Note that the
        greater the sample rate, the greater the noise. For multiple channels,
        the achieved sample rate is roughly the target sample rate divided by
        the number of channels.
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        strings representing the channels to acquire. Each channel must follow
        one of the two syntax :
        ::

          'INi', i in range(10)

        or else
        ::

          'INi-INj', i, j in range(10)

        With the first syntax, the output is the voltage difference between the
        acquired channel and the ground of the power supply. With the second
        syntax, the output value is simply the difference between the voltages
        of the channels `i` and `j`. It is preferable to use the channels in
        pair (0 & 1, 2 & 3, etc.) for differential acquisition. The data from
        the different channels is acquired sequentially, not all at once.
      digital_filter: The Waveshare Hat features a digital filter that can
        accept different settings. Refer to the documentation of the ADS1263
        for more detail.
      gain: Allows to tune the output values of the ADC according to the
        formula : :math:`output = gain * tension + offset`. The same gain
        applies to all the channels.
      offset: Allows to tune the output values of the ADC according to the
        formula : :math:`output = gain * tension + offset`. The same offset
        applies to all the channels.

    Important:
      When the ``gain_hardware`` is greater than 1, the PGA cannot amplify
      above 4.7V or under 0.3V. For example a 2.8V signal read with a gain of 2
      would be read as 4.7V after the PGA, not 4.8V ! Beware !
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._bus = None

    super().__init__()
    self._gain = gain
    self._offset = offset

    self.log(logging.INFO, f"Opening the SPI communication on port {spi_port}")
    self._bus = SpiDev(spi_port, 0)

    # Checking the validity of the arguments
    if gain_hardware not in ADS1263_GAIN:
      raise ValueError(f"gain should be in {list(ADS1263_GAIN)}")
    self._gain_hardware = gain_hardware

    if sample_rate not in ADS1263_DRATE:
      raise ValueError(f'sample rate should be in {list(ADS1263_DRATE)}')
    self._sample_rate = sample_rate

    if digital_filter not in range(5):
      raise ValueError(f'digital_filter should be in {list(range(5))}')
    self._filter = digital_filter

    # Parsing the channels to check the right syntax was given
    if channels is None:
      channels = ['IN0']
    else:
      channels = list(channels)

    for channel in channels:
      if fullmatch(r'IN\d', channel) is None and \
            fullmatch(r'IN\d\s?-\s?IN\d', channel) is None:
        raise ValueError("Valid formats for adc_channels values are "
                         "either 'INi' (i in range(10)) or 'INi - INj'")
    self._chan = channels

  def open(self) -> None:
    """Sets up the GPIO and the different parameters on the ADS1263."""

    # Setting up the GPIO
    self.log(logging.INFO, "Setting up the GPIOs")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RST_PIN, GPIO.OUT)
    GPIO.setup(CS_PIN, GPIO.OUT)
    GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Setting up the SPI bus
    self.log(logging.INFO, "Setting up the SPI connection")
    self._bus.max_speed_hz = 2000000
    self._bus.mode = 0b01

    # Resetting the ADS1263
    self.log(logging.INFO, "Configuring the ADS1263")
    GPIO.output(RST_PIN, GPIO.HIGH)
    sleep(0.2)
    GPIO.output(RST_PIN, GPIO.LOW)
    sleep(0.2)
    GPIO.output(RST_PIN, GPIO.HIGH)
    sleep(0.2)

    # Un-raising the reset flag
    self._write_reg(ADS1263_REG['REG_POWER'], 0x01)

    # Applying the sample rate and gain settings
    self._write_cmd(ADS1263_CMD['CMD_STOP1'])
    self._write_reg(ADS1263_REG['REG_MODE2'],
                    ADS1263_DRATE[self._sample_rate] |
                    ADS1263_GAIN[self._gain_hardware] << 4)

    # Setting the 5V supply as voltage reference
    self._write_reg(ADS1263_REG['REG_REFMUX'], 0x24)
    # Setting the first channel
    self._set_channel(self._chan[0])
    # Setting the filter
    self._write_reg(ADS1263_REG['REG_MODE1'], self._filter << 5)
    # Disabling the status and checksum bits
    self._write_reg(ADS1263_REG['REG_INTERFACE'], 0x00)

    # Starting the acquisition
    self._write_cmd(ADS1263_CMD['CMD_START1'])

  def get_data(self) -> list[float]:
    """Reads the channels sequentially, and returns all the values along with a
    timestamp.

    Time is returned first, and the values are then returned in the same order
    as the channels were given.
    """

    out = [time()]

    # Reading the channels sequentially
    for channel in self._chan:

      # Don't bother setting the channel if it never changes !
      if len(self._chan) > 1:
        self._set_channel(channel)

      # Waiting for a conversion to complete
      self._wait_drdy()

      # Reading the data from the buffer
      GPIO.output(CS_PIN, GPIO.LOW)
      self.log(logging.DEBUG, f"Writing {[ADS1263_CMD['CMD_RDATA1']]} to the "
                              f"SPI bus")
      self._bus.writebytes([ADS1263_CMD['CMD_RDATA1']])
      buf = self._bus.readbytes(4)
      self.log(logging.DEBUG, f"Read {buf} from the SPI bus")
      GPIO.output(CS_PIN, GPIO.HIGH)

      # Assembling the data from the four registers
      read = (buf[0] << 24) & 0xFF000000
      read |= (buf[1] << 16) & 0xFF0000
      read |= (buf[2] << 8) & 0xFF00
      read |= (buf[3]) & 0xFF

      # Converting to Volts
      if read >> 31 & 0b1:
        volt = (read / 0x80000000 - 2) * VREF / self._gain_hardware
      else:
        volt = read / 0x7FFFFFFF * VREF / self._gain_hardware

      out.append(self._gain * volt + self._offset)

    return out

  def close(self) -> None:
    """Closes the SPI bus and resets the GPIOs."""

    if self._bus is not None:
      self.log(logging.INFO, "Closing the SPI communication")
      self._bus.close()
    self.log(logging.INFO, "Cleaning up the GPIOs")
    GPIO.cleanup()

  def _write_cmd(self, cmd: int) -> None:
    """Writes a command to the ADS1263 and manages the CS pin."""

    GPIO.output(CS_PIN, GPIO.LOW)
    self.log(logging.DEBUG, f"Writing the command {cmd} to the SPI bus")
    self._bus.writebytes([cmd])
    GPIO.output(CS_PIN, GPIO.HIGH)

  @staticmethod
  def _wait_drdy() -> None:
    """Waits for the DRDY_PIN to be low and returns, or raises a
    :exc:`TimeoutError` if data wasn't available within 0.5s."""

    t0 = time()
    while time() - t0 < 0.5:
      if not GPIO.input(DRDY_PIN):
        return
    raise TimeoutError

  def _write_reg(self, reg: int, data: int) -> None:
    """Writes data to a register of the ADS1263 and manages the CS pin."""

    GPIO.output(CS_PIN, GPIO.LOW)
    cmd = [ADS1263_CMD['CMD_WREG'] | reg, 0x00, data]
    self.log(logging.DEBUG, f"Writing the data {cmd} to the SPI bus")
    self._bus.writebytes(cmd)
    GPIO.output(CS_PIN, GPIO.HIGH)

  def _set_channel(self, channel: str) -> None:
    """Parses the channel string to set, and writes the corresponding values to
    the ADS1263."""

    # Parsing the channel
    try:
      pos_chan, neg_chan = findall(r'\d', channel)
    except ValueError:
      pos_chan, *_ = findall(r'\d', channel)
      neg_chan = 10

    # Setting it
    self._write_reg(ADS1263_REG['REG_INPMUX'],
                    int(pos_chan) << 4 | int(neg_chan))
