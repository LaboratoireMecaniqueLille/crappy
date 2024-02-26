# coding: utf-8

from time import time

from .meta_actuator import Actuator


class FakeDCMotor(Actuator):
  """Emulates the behavior of a DC electric machine, driven through its input
  voltage.

  It is mainly intended for testing scripts without requiring any hardware.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 Renamed from *Fake_motor* to *FakeDCMotor*
  """

  def __init__(self,
               inertia: float = 0.5,
               torque: float = 0.,
               kv: float = 1000,
               rv: float = 0.4,
               fv: float = 2e-5,
               simulation_speed: float = 1,
               initial_speed: float = 0,
               initial_pos: float = 0) -> None:
    """Sets the instance attributes.

    Args:
      inertia: The inertia of the motor, in `kg.m²`.
      torque: A constant torque applied on the shaft of the motor in `N.m`.
      kv: The electrical constant of the motor, in`t/min/V`.
      rv: The internal solid friction coefficient of the motor, no unit.
      fv: The internal fluid friction coefficient of the motor, no unit.
      simulation_speed: Speed factor of the simulation, to speed it up or slow
        it down.

        .. versionchanged:: 2.0.0
           renamed from *sim_speed* to *simulation_speed*
      initial_speed: The initial speed of the motor, in RPM.
      initial_pos: The initial position of the motor, in turns.
    """

    super().__init__()

    self._inertia = inertia
    self._torque = torque
    self._kv = kv
    self._rv = rv
    self._fv = fv
    self._simulation_speed = simulation_speed
    self._initial_speed = initial_speed
    self._initial_pos = initial_pos

    self._rpm = 0
    self._pos = 0
    self._volt = 0
    self._t = time()

  def open(self) -> None:
    """Sets the variables describing the state of the motor."""

    self._rpm = self._initial_speed
    self._pos = self._initial_pos
    self._volt = 0
    self._t = time() * self._simulation_speed

  def get_speed(self) -> float:
    """Return the speed of the motor, in RPM."""

    self._update()
    return self._rpm

  def get_position(self) -> float:
    """Returns the position of the motor, in rounds.

    .. versionchanged:: 1.5.2 renamed from *get_pos* to *get_position*
    """

    self._update()
    return self._pos

  def set_speed(self, volt: float) -> None:
    """Sets the command of the motor, in volts."""

    self._update()
    self._volt = volt

  def _update(self) -> None:
    """Updates the motor variables based on the timestamp and their previous
    values.

    It supposes that the voltage has been constant since the last update.
    """

    t1 = time() * self._simulation_speed
    dt = (t1 - self._t)
    self._t = t1

    f = self._volt * self._kv - self._torque - self._rpm * \
        (1 + self._rv + self._rpm * self._fv)
    drpm = f / self._inertia * dt
    self._pos += dt * (self._rpm + drpm / 2)
    self._rpm += drpm
