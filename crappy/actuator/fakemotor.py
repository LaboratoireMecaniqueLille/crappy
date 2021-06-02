# coding: utf-8

from time import time

from .actuator import Actuator


class Fake_motor(Actuator):
  """
  To run test programs without a physical actuator.

  Note:
    A virtual motor driven by a voltage, you can set its properties with the
    args. It has the same methods as a real motor: open, set_speed, get_speed,
    get_pos.

  Args:
    - inertia (float, default: .5): Inertia of the motor.
    - torque (float, default: 0): A torque applied on the axis.
    - kv (float, default: 1000): The electrical constant of the motor.
    - rv (float, default: .4): The solid friction.
    - fv (float, default: 2e-5): The fluid friction.

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
               **kwargs):
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

  def open(self):
    self.rpm = self.initial_speed
    self.pos = self.initial_pos
    self.u = 0  # V
    self.t = time() * self.sim_speed

  def stop(self):
    self.set_speed(0)

  def close(self):
    pass

  def update(self):
    """
    Will update the motor rpm.

    Note:
      Supposes u is constant for the interval dt.

    """

    t1 = time() * self.sim_speed
    dt = (t1 - self.t)
    self.t = t1
    f = self.u * self.kv - self.torque - self.rpm * \
        (1 + self.rv + self.rpm * self.fv)
    drpm = f / self.inertia * dt
    self.pos += dt * (self.rpm + drpm / 2)
    self.rpm += drpm

  def get_speed(self):
    """
    Return the motor speed (rpm).
    """

    self.update()
    return self.rpm

  def get_pos(self):
    """
    Returns the motor position.
    """

    self.update()
    return self.pos

  def set_speed(self, u):
    """
    Sets the motor cmd in volts.
    """

    self.update()
    self.u = u
