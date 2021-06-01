# coding: utf-8

from __future__ import print_function

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

  def __init__(self, **kwargs):
    super().__init__()
    for arg, default in [("inertia", .5),  # Inertia of the motor
                        ("torque", 0),  # Counter torque on the axis
                        ("kv", 1000),  # t/min/V
                        ("rv", .4),  # Solid friction
                        ("fv", 2e-5),  # Fluid friction
                        ("sim_speed", 1),  # Speed factor of the simulation
                        ("initial_speed", 0),
                        ("initial_pos", 0), ]:
      setattr(self, arg, kwargs.pop(arg, default))
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
