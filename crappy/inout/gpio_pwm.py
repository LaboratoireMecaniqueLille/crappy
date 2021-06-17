# coding: utf-8

from .inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")


class Gpio_pwm(InOut):
  """ Class for performing PWM on a Raspberry Pi.

  The Gpio_pwn InOut block is meant for controlling a PWM output from a
  Raspberry Pi GPIO. It allows to control the duty cycle, the frequency, or
  both at the same time. When controlling both, the duty cycle should be first
  and the frequency second in the list of inputs.

  Warning:
    Only works on Raspberry Pi !
  """

  def __init__(self,
               pin_out: int,
               duty_cycle: float = None,
               frequency: float = None) -> None:
    """Checks the arguments validity.

    Args:
      pin_out (:obj:`int`): The GPIO pin to be controlled (BCM convention).
      duty_cycle (:obj:`float`): If provided, sets a fixed duty cycle for the
        entire assay. Only the frequency can then be tuned. If not provided,
        the block will expect the duty cycle values to be given as the first
        input. It will also start the PWM with a duty cycle of `0%` before the
        first value is received and set.
      frequency (:obj:`float`): If provided, sets a fixed PWM frequency for the
        entire assay. Only the duty cycle can then be tuned. If not provided,
        the block will expect the frequency values to be given as the first
        input if the ``duty_cycle`` argument is provided, or else as the second
        input. It will also start the PWM with a frequency of `10kHz` before
        the first value is received and set.

    Note:
      - ``duty_cycle``:
        The duty cycle inputs are clamped between `0` and `100`.

      - ``frequency``:
        The frequency inputs are clamped between `10Hz` and `1MhZ`. Sending
        other values to the bloc doesn't raise any error, but the assay may not
        run as expected.

      - **Hardware PWM pins**:
        On the Raspberry Pi 4, only the GPIO pins `12`, `13`, `18` and `19`
        support hardware PWM. Trying to get a PWM output from other pins might
        work but may decrease the available frequency range.
    """

    InOut.__init__(self)

    if pin_out not in range(2, 28):
      raise ValueError("pin_out should be an integer between 2 and 28")
    else:
      self._pin_out = pin_out

    if frequency is not None:
      if not 10 <= frequency < 1000000:
        raise ValueError("frequency should be between 100Hz and 1MHz")
    self._frequency = frequency

    if duty_cycle is not None:
      if not 0 <= duty_cycle <= 100:
        raise ValueError("Duty cycle should be positive and not exceed 100%")
    self._duty_cycle = duty_cycle

  def open(self) -> None:
    """Sets the GPIOs and starts the PWM."""

    # Setting the GPIOs
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(self._pin_out, GPIO.OUT)

    # Setting to user frequency if provided, or else to 10kHz
    if self._frequency is not None:
      self._pwm = GPIO.PWM(self._pin_out, self._frequency)
    else:
      self._pwm = GPIO.PWM(self._pin_out, 10000)

    # Setting to user duty cycle if provided, or else to 0%
    if self._duty_cycle is not None:
      self._pwm.start(self._duty_cycle)
    else:
      self._pwm.start(0)

  def set_cmd(self, *cmd: float) -> None:
    """Modifies the PWM frequency and/or duty cycle.

    Args:
      *cmd (:obj:`float`): Values of duty cycle and/or frequency to set
    """

    cmd = list(*cmd)
    # If both frequency and duty cycle are fixed by the user, nothing to do
    if self._duty_cycle is not None and self._frequency is not None:
      pass

    # If only frequency is fixed, setting the duty cycle
    elif self._frequency is not None:
      dc = min(100, max(0, cmd[0]))
      self._pwm.ChangeDutyCycle(dc)

    # If only the duty cycle is fixed, setting the frequency
    elif self._duty_cycle is not None:
      freq = min(1000000, max(10, cmd[0]))
      self._pwm.ChangeFrequency(freq)

    # If neither duty cycle nor frequency are fixed, setting both
    else:
      dc = min(100, max(0, cmd[0]))
      freq = min(1000000, max(10, cmd[1]))
      self._pwm.ChangeFrequency(freq)
      self._pwm.ChangeDutyCycle(dc)

  def close(self) -> None:
    """Stops PWM and releases GPIOs."""

    self._pwm.stop()
    GPIO.cleanup()
