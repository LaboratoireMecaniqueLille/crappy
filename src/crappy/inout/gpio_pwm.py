# coding: utf-8

from typing import Optional
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")


class GPIOPWM(InOut):
  """This class can drive a PWM output on a Raspberry Pi.

  It allows controlling the duty cycle, the frequency, or both at the same
  time. When controlling both, the duty cycle should be first and the frequency
  second in the given commands.

  Warning:
    Only works on Raspberry Pi !
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Gpio_pwm* to *GPIOPWM*
  """

  def __init__(self,
               pin_out: int,
               duty_cycle: Optional[float] = None,
               frequency: Optional[float] = None) -> None:
    """Checks the validity of the arguments.

    Args:
      pin_out: The index of GPIO pin to drive in BCM convention, as an
        :obj:`int`.
      duty_cycle: If provided (as a :obj:`float`, in percent), sets a fixed
        duty cycle for the entire test. Only the frequency can then be tuned.
        If not provided, the duty cycle can be set as a command. The duty cycle
        will also be set to `0%` until a first value is received.
      frequency: If provided (as a :obj:`float`, in Hz), sets a fixed PWM
        frequency for the entire test. Only the duty cycle can then be tuned.
        If not provided, the frequency can be set as a command. The frequency
        will also be set to `10kHz` until a first value is received. Note that
        the frequency inputs are clamped between `10Hz` and `1MHz`.

    Note:
      Several values can be passed at once as a command. If both ``duty_cycle``
      and ``frequency`` are provided, all the values are ignored. If only
      ``frequency`` is provided, the first command value sets the duty cycle
      and any other value is ignored. Same goes if only ``duty_cycle`` is
      provided. If none of the two arguments are provided, the first command
      value should set the duty cycle and the second command value should set
      the frequency.

    Note:
      On the Raspberry Pi 4, only the GPIO pins `12`, `13`, `18` and `19`
      support hardware PWM. Trying to get a PWM output from other pins might
      work but may decrease the available frequency range.
    """

    self._pwm = None

    super().__init__()

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
    self.log(logging.INFO, "Setting up the GPIOs")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(self._pin_out, GPIO.OUT)

    # Setting to user frequency if provided, or else to 10kHz
    self.log(logging.INFO, "Setting up the PWM")
    if self._frequency is not None:
      self._pwm = GPIO.PWM(self._pin_out, self._frequency)
    else:
      self._pwm = GPIO.PWM(self._pin_out, 10000)

    # Setting to user duty cycle if provided, or else to 0%
    self.log(logging.INFO, "Starting the PWM")
    if self._duty_cycle is not None:
      self._pwm.start(self._duty_cycle)
    else:
      self._pwm.start(0)

  def set_cmd(self, *cmd: float) -> None:
    """Modifies the PWM frequency and/or duty cycle.

    Args:
      *cmd: Values of duty cycle and/or frequency to set.
    """

    # If both frequency and duty cycle are fixed by the user, nothing to do
    if self._duty_cycle is not None and self._frequency is not None:
      return

    # If only frequency is fixed, setting the duty cycle
    elif self._frequency is not None:
      dc = min(100., max(0., cmd[0]))
      self._pwm.ChangeDutyCycle(dc)

    # If only the duty cycle is fixed, setting the frequency
    elif self._duty_cycle is not None:
      freq = min(1000000., max(10., cmd[0]))
      self._pwm.ChangeFrequency(freq)

    # If neither duty cycle nor frequency are fixed, setting both
    else:
      dc = min(100., max(0., cmd[0]))
      freq = min(1000000., max(10., cmd[1]))
      self._pwm.ChangeFrequency(freq)
      self._pwm.ChangeDutyCycle(dc)

  def close(self) -> None:
    """Stops the PWM and releases the GPIOs."""

    if self._pwm is not None:
      self.log(logging.INFO, "Stopping the PWM")
      self._pwm.stop()

    self.log(logging.INFO, "Cleaning up the GPIOs")
    GPIO.cleanup()
