# coding: utf-8

import logging
from typing import Optional

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  from Phidget22.Net import Net, PhidgetServerType
  from Phidget22.Devices.Stepper import Stepper, StepperControlMode
  from Phidget22.PhidgetException import PhidgetException
except (ImportError, ModuleNotFoundError):
  Net = OptionalModule('Phidget22')
  PhidgetServerType = OptionalModule('Phidget22')
  Stepper = OptionalModule('Phidget22')
  StepperControlMode = OptionalModule('Phidget22')
  PhidgetException = OptionalModule('Phidget22')


class Phidget4AStepper(Actuator):
  """This class can drive Phidget's 4A Stepper module in speed or in position.

  It relies on the :mod:`Phidget22` module to communicate with the motor
  driver. The driver can deliver up to 4A to the motor, and uses 16 microsteps
  by default. Its acquisition rate is set to 10 values per second in this
  class.

  The distance unit is the `mm` and the time unit is the `s`, so speeds are in
  `mm/s` and accelerations in `mm/s²`.
  """

  def __init__(self,
               steps_per_mm: float,
               current_limit: float,
               max_acceleration: Optional[float] = None,
               remote: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      steps_per_mm: The number of steps necessary to move by `1 mm`. This value
        is used to set a conversion factor on the driver, so that it can be
        driven in `mm` and `s` units.
      current_limit: The maximum current the driver is allowed to deliver to
        the motor, in `A` .
      max_acceleration: If given, sets the maximum acceleration the motor is
        allowed to reach in `mm/s²`.
    """

    self._motor: Optional[Stepper] = None

    super().__init__()

    self._steps_per_mm = steps_per_mm
    self._current_limit = current_limit
    self._max_acceleration = max_acceleration
    self.remote = remote

    # These buffers store the last known position and speed
    self._last_velocity: Optional[float] = None
    self._last_position: Optional[float] = None

  def open(self) -> None:
    """Sets up the connection to the motor driver as well as the various
    callbacks, and waits for the motor driver to attach."""

    # Setting up the motor driver
    self.log(logging.DEBUG, "Enabling server discovery")
    Net.enableServerDiscovery(PhidgetServerType.PHIDGETSERVER_DEVICEREMOTE)
    self._motor = Stepper()
    if self.remote is True:
      self._motor.setIsLocal(False)
      self._motor.setIsRemote(True)
    else:
      self._motor.setIsLocal(True)
      self._motor.setIsRemote(False)
    self._motor.setIsRemote(True)

    # Setting up the callbacks
    self.log(logging.DEBUG, "Setting the callbacks")
    self._motor.setOnAttachHandler(self._on_attach)
    self._motor.setOnErrorHandler(self._on_error)
    self._motor.setOnVelocityChangeHandler(self._on_velocity_change)
    self._motor.setOnPositionChangeHandler(self._on_position_change)

    # Opening the connection to the motor driver
    try:
      self.log(logging.DEBUG, "Trying to attach the motor")
      self._motor.openWaitForAttachment(10000)
    except PhidgetException:
      raise TimeoutError("Waited too long for the motor to attach !")

    # Energizing the motor
    self._motor.setEngaged(True)

  def set_speed(self, speed: float) -> None:
    """Sets the requested speed for the motor.

    Switches to the correct driving mode if needed.

    Args:
      speed: The speed to reach, in `mm/s`.
    """

    # Switching the control mode if needed
    if not self._motor.getControlMode() == StepperControlMode.CONTROL_MODE_RUN:
      self.log(logging.DEBUG, "Setting the control mode to run")
      self._motor.setControlMode(StepperControlMode.CONTROL_MODE_RUN)

    # Setting the desired velocity
    if abs(speed) > self._motor.getMaxVelocityLimit():
      raise ValueError(f"Cannot set a velocity greater than "
                       f"{self._motor.getMaxVelocityLimit()} mm/s !")
    else:
      self._motor.setVelocityLimit(speed)

  def set_position(self,
                   position: float,
                   speed: Optional[float] = None) -> None:
    """Sets the requested position for the motor.

    Switches to the correct driving mode if needed.

    Args:
      position: The position to reach, in `mm`.
      speed: If not :obj:`None`, the speed to use for moving to the desired
        position.
    """

    # Switching the control mode if needed
    if not (self._motor.getControlMode() ==
            StepperControlMode.CONTROL_MODE_STEP):
      self.log(logging.DEBUG, "Setting the control mode to step")
      self._motor.setControlMode(StepperControlMode.CONTROL_MODE_STEP)

    # Setting the desired velocity if required
    if speed is not None:
      if abs(speed) > self._motor.getMaxVelocityLimit():
        raise ValueError(f"Cannot set a velocity greater than "
                         f"{self._motor.getMaxVelocityLimit()} mm/s !")
      else:
        self._motor.setVelocityLimit(abs(speed))

    # Setting the requested position
    min_pos = self._motor.getMinPosition()
    max_pos = self._motor.getMaxPosition()
    if not min_pos <= position <= max_pos:
      raise ValueError(f"The position value must be between {min_pos} and "
                       f"{max_pos}, got {position} !")
    else:
      self._motor.setTargetPosition(position)

  def get_speed(self) -> Optional[float]:
    """Returns the last known speed of the motor."""

    return self._last_velocity

  def get_position(self) -> Optional[float]:
    """Returns the last known position of the motor."""

    return self._last_position

  def stop(self) -> None:
    """Deenergizes the motor."""

    if self._motor is not None:
      self._motor.setEngaged(False)

  def close(self) -> None:
    """Closes the connection to the motor."""

    if self._motor is not None:
      self._motor.close()

  def _on_attach(self, _: Stepper) -> None:
    """Callback called when the motor driver attaches to the program.

    It sets the current limit, scale factor, data rate and maximum
    acceleration.
    """

    self.log(logging.INFO, "Motor successfully attached")

    # Setting the current limit for the motor
    min_current = self._motor.getMinCurrentLimit()
    max_current = self._motor.getMaxCurrentLimit()
    if not min_current <= self._current_limit <= max_current:
      raise ValueError(f"The current limit should be between {min_current} $"
                       f"and {max_current} A !")
    else:
      self._motor.setCurrentLimit(self._current_limit)

    # Setting the scale factor and the data rate
    self._motor.setRescaleFactor(1 / 16 / self._steps_per_mm)
    self._motor.setDataInterval(100)

    # Setting the maximum acceleration
    if self._max_acceleration is not None:
      min_accel = self._motor.getMinAcceleration()
      max_accel = self._motor.getMaxAcceleration()
      if not min_accel <= self._max_acceleration <= max_accel:
        raise ValueError(f"The maximum acceleration should be between "
                         f"{min_accel} and {max_accel} m/s² !")
      else:
        self._motor.setAcceleration(self._max_acceleration)

  def _on_error(self, _: Stepper, error_code: int, error: str) -> None:
    """Callback called when the motor driver returns an error."""

    raise RuntimeError(f"Got error with error code {error_code}: {error}")

  def _on_velocity_change(self, _: Stepper, velocity: float) -> None:
    """Callback called when the motor velocity changes."""

    self.log(logging.DEBUG, f"Velocity changed to {velocity}")
    self._last_velocity = velocity

  def _on_position_change(self, _: Stepper, position: float) -> None:
    """Callback called when the motor position changes."""

    self.log(logging.DEBUG, f"Position changed to {position}")
    self._last_position = position
