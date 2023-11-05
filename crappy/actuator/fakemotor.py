# coding: utf-8

from time import time
from warnings import warn

from .actuator import Actuator


class Fake_motor(Actuator):
  """To run test programs without a physical actuator.

  Note:
    A virtual motor driven by a voltage, you can set its properties with the
    args. It has the same methods as a real motor: :meth:`open`,
    :meth:`set_speed`, :meth:`get_speed`, :meth:`get_position`.
  """

  def __init__(self,
               inertia: float = 0.5,
               torque: float = 0.,
               kv: float = 1000,
               rv: float = 0.4,
               fv: float = 2e-5,
               sim_speed: float = 1,
               initial_speed: float = 0,
               initial_pos: float = 0,
               **kwargs) -> None:
    """Sets the instance attributes.

    Args:
      inertia (:obj:`float`, optional): Inertia of the motor (`kg.m²`).
      torque(:obj:`float`, optional): A torque applied on the axis (`N.m`).
      kv(:obj:`float`, optional): The electrical constant of the motor
        (`t/min/V`).
      rv(:obj:`float`, optional): The solid friction.
      fv(:obj:`float`, optional): The fluid friction.
      sim_speed(:obj:`float`, optional): Speed factor of the simulation.
      initial_speed(:obj:`float`, optional): (`rpm`)
      initial_pos(:obj:`float`, optional): (turns)
    """

    warn("The Fake_motor Actuator will be renamed to FakeDCMotor in version "
         "2.0.0", FutureWarning)

    if sim_speed != 1:
      warn("The sim_speed argument will be renamed to simulation_speed in "
           "version 2.0.0", FutureWarning)

    super().__init__()
    self.inertia = inertia
    self.torque = torque
    self.kv = kv
    self.rv = rv
    self.fv = fv
    self.sim_speed = sim_speed
    self.initial_speed = initial_speed
    self.initial_pos = initial_pos
    assert not kwargs, "Fake_motor got invalid kwarg(s): " + str(kwargs)

  def open(self) -> None:
    self.rpm = self.initial_speed
    self.pos = self.initial_pos
    self.u = 0  # V
    self.t = time() * self.sim_speed

  def stop(self) -> None:
    self.set_speed(0)

  def close(self) -> None:
    pass

  def update(self) -> None:
    """Updates the motor rpm.

    Note:
      Supposes `u` is constant for the interval `dt`.
    """

    warn("The update method will be renamed to _update in version 2.0.0",
         FutureWarning)

    t1 = time() * self.sim_speed
    dt = (t1 - self.t)
    self.t = t1
    f = self.u * self.kv - self.torque - self.rpm * \
        (1 + self.rv + self.rpm * self.fv)
    drpm = f / self.inertia * dt
    self.pos += dt * (self.rpm + drpm / 2)
    self.rpm += drpm

  def get_speed(self) -> float:
    """Return the motor speed (rpm)."""

    self.update()
    return self.rpm

  def get_position(self) -> float:
    """Returns the motor position."""

    self.update()
    return self.pos

  def set_speed(self, u: float) -> None:
    """Sets the motor `cmd` in volts."""

    self.update()
    self.u = u
