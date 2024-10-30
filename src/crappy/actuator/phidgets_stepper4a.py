# coding: utf-8

import logging
import numpy as np
from pathlib import Path
from platform import system
from typing import Optional, Union

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  from Phidget22.Net import Net, PhidgetServerType
  from Phidget22.Devices.Stepper import Stepper, StepperControlMode
  from Phidget22.Devices.DigitalInput import DigitalInput
  from Phidget22.PhidgetException import PhidgetException
except (ImportError, ModuleNotFoundError):
  Net = OptionalModule('Phidget22')
  PhidgetServerType = OptionalModule('Phidget22')
  Stepper = OptionalModule('Phidget22')
  StepperControlMode = OptionalModule('Phidget22')
  DigitalInput = OptionalModule('Phidget22')
  PhidgetException = OptionalModule('Phidget22')


class Phidget4AStepper(Actuator):
  """This class can drive Phidget's 4A Stepper module in speed or in position.

  It relies on the :mod:`Phidget22` module to communicate with the motor
  driver. The driver can deliver up to 4A to the motor, and uses 16 microsteps
  by default. Its acquisition rate is set to 10 values per second in this
  class.

  The distance unit is the `mm` and the time unit is the `s`, so speeds are in
  `mm/s` and accelerations in `mm/s²`.

  .. versionadded:: 2.0.3
  """

  def __init__(self,
               steps_per_mm: float,
               current_limit: float,
               max_acceleration: Optional[float] = None,
               remote: bool = False,
               absolute_mode: bool = False,
               reference_pos: float = 0,
               switch_ports: tuple[int, ...] = tuple(),
               save_last_pos: bool = False,
               save_pos_folder: Optional[Union[str, Path]] = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      steps_per_mm: The number of steps necessary to move by `1 mm`. This value
        is used to set a conversion factor on the driver, so that it can be
        driven in `mm` and `s` units.
      current_limit: The maximum current the driver is allowed to deliver to
        the motor, in `A` .
      max_acceleration: If given, sets the maximum acceleration the motor is
        allowed to reach in `mm/s²`.
      remote: Set to :obj:`True` to drive the stepper via a network VINT Hub,
        or to :obj:`False` to drive it via a USB VINT Hub.
      absolute_mode: If :obj:`True`, the target position of the motor will be
        calculated from a reference position. If :obj:`False`, the target
        position of the motor will be calculated from its current position.

        .. versionadded:: 2.0.4
      reference_pos: The position considered as the reference position at the
        beginning of the test. Only takes effect if ``absolute_mode`` is
        :obj:`True`.

        .. versionadded:: 2.0.4
      switch_ports: The indexes of the VINT Hub ports where the switches are
        connected.

        .. versionadded:: 2.0.4
      save_last_pos: If :obj:`True`, the last position of the actuator will be
        saved in a .npy file.

        .. versionadded:: 2.0.4
      save_pos_folder: The path to the folder where to save the last position
        of the motor. Only takes effect if ``save_last_pos`` is :obj:`True`.

        .. versionadded:: 2.0.4
    """

    self._motor: Optional[Stepper] = None

    super().__init__()

    self._steps_per_mm = steps_per_mm
    self._current_limit = current_limit
    self._max_acceleration = max_acceleration
    self._remote = remote
    self._switch_ports = switch_ports
    self._switches = list()

    # The following attribute is set to True to automatically check the state
    # of the switches in the open method to keep the motor to move if a switch
    # has been disconnected or hit.
    # Nevertheless, this check can be bypassed if the motor is used outside
    # from a Crappy loop by setting the attribute to False.
    self._check_switch = True

    self._absolute_mode = absolute_mode
    if self._absolute_mode:
      self._ref_pos = reference_pos

    # Determining the path where to save the last position
    # It depends on the current operating system
    self._path: Optional[Path] = None
    if save_last_pos:
      if save_pos_folder is not None:
        self._path = Path(save_pos_folder)
      elif system() in ('Linux', 'Darwin'):
        self._path = Path.home() / '.machine1000N'
      elif system() == 'Windows':
        self._path = Path.home() / 'AppData' / 'Local' / 'machine1000N'
      else:
        self._save_last_pos = False

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

    # Setting up the switches
    for port in self._switch_ports:
      switch = DigitalInput()
      switch.setIsHubPortDevice(True)
      switch.setHubPort(port)
      self._switches.append(switch)

    # Setting the remote or local status
    self._motor.setIsLocal(not self._remote)
    self._motor.setIsRemote(self._remote)
    for switch in self._switches:
      switch.setIsLocal(not self._remote)
      switch.setIsRemote(self._remote)

    # Setting up the callbacks
    self.log(logging.DEBUG, "Setting the callbacks")
    self._motor.setOnAttachHandler(self._on_attach)
    self._motor.setOnErrorHandler(self._on_error)
    self._motor.setOnVelocityChangeHandler(self._on_velocity_change)
    self._motor.setOnPositionChangeHandler(self._on_position_change)
    for switch in self._switches:
      switch.setOnStateChangeHandler(self._on_end)

    # Opening the connection to the motor driver
    try:
      self.log(logging.DEBUG, "Trying to attach the motor")
      self._motor.openWaitForAttachment(10000)
    except PhidgetException:
      raise TimeoutError("Waited too long for the motor to attach !")

    # Opening the connection to the switches
    for switch in self._switches:
      try:
        self.log(logging.DEBUG, "Trying to attach the switch")
        switch.openWaitForAttachment(10000)
      except PhidgetException:
        raise TimeoutError("Waited too long for the switch to attach !")

    # Energizing the motor
    self._motor.setEngaged(True)

    # Check the state of the switches
    if self._check_switch and not all(
      switch.getState() for switch in self._switches):
      raise ValueError(f"A switch is already hit or disconnected !")

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

    # Ensuring that the requested position is valid
    min_pos = self._motor.getMinPosition()
    max_pos = self._motor.getMaxPosition()
    if not min_pos <= position <= max_pos:
      raise ValueError(f"The position value must be between {min_pos} and "
                       f"{max_pos}, got {position} !")

    # Setting the position depending on the driving mode
    if not self._absolute_mode:
      self._motor.setTargetPosition(position)
    else:
      self._motor.setTargetPosition(position - self._ref_pos)

  def get_speed(self) -> Optional[float]:
    """Returns the last known speed of the motor."""

    return self._last_velocity

  def get_position(self) -> Optional[float]:
    """Returns the last known position of the motor."""

    if not self._absolute_mode:
      return self._last_position

    # Cannot perform addition if no known last position
    elif self._last_position is not None:
      return self._last_position + self._ref_pos

  def stop(self) -> None:
    """Deenergizes the motor."""

    if self._motor is not None:
      self._motor.setEngaged(False)

  def close(self) -> None:
    """Closes the connection to the motor."""

    if self._motor is not None:
      if self._path is not None:
        self._path.mkdir(parents=False, exist_ok=True)
        np.save(self._path / 'last_pos.npy', self.get_position())
      self._motor.close()

    for switch in self._switches:
      switch.close()

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

  def _on_end(self, _: DigitalInput, state) -> None:
    """Callback when a switch is hit."""

    if not bool(state):
      self.stop()
      raise ValueError(f"A switch has been hit or disconnected !")
