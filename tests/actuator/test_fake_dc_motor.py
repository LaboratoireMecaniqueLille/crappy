# coding: utf-8

from unittest import TestCase
from unittest.mock import patch
from crappy.actuator.fake_dc_motor import FakeDCMotor
import crappy.actuator.fake_dc_motor as fake_dc_motor_module


class TestFakeDCMotor(TestCase):
  """Unit tests for the FakeDCMotor Actuator."""

  def test_inertia_cannot_be_zero(self) -> None:
    """Checks that physically invalid inertia is rejected."""

    with self.assertRaises(ValueError):
      FakeDCMotor(inertia=0)

  def test_negative_inertia_is_normalized(self) -> None:
    """Checks that negative inertia values are accepted as magnitudes."""

    motor = FakeDCMotor(inertia=-2)

    self.assertEqual(motor._inertia, 2)

  def test_open_initializes_state_from_constructor_arguments(self) -> None:
    """Checks that open resets the simulated motor state."""

    with patch.object(fake_dc_motor_module,
                      'time',
                      side_effect=[1.0, 2.0]):
      motor = FakeDCMotor(initial_speed=12,
                          initial_pos=3,
                          simulation_speed=4)
      motor._rpm = 99
      motor._pos = 99
      motor._volt = 99

      motor.open()

    self.assertEqual(motor._rpm, 12)
    self.assertEqual(motor._pos, 3)
    self.assertEqual(motor._volt, 0)
    self.assertEqual(motor._t, 8)

  def test_set_speed_updates_state_before_storing_voltage(self) -> None:
    """Checks that a voltage command starts from the current simulated time."""

    motor = FakeDCMotor(inertia=1, kv=0, rv=-1, fv=0)
    motor._rpm = 60
    motor._pos = 0
    motor._volt = 0
    motor._t = 0

    with patch.object(fake_dc_motor_module, 'time', return_value=1):
      motor.set_speed(5)

    self.assertAlmostEqual(motor._rpm, 60)
    self.assertAlmostEqual(motor._pos, 1)
    self.assertEqual(motor._volt, 5)
    self.assertEqual(motor._t, 1)

  def test_get_speed_updates_rpm_and_position(self) -> None:
    """Checks the fake motor dynamics when reading the speed."""

    motor = FakeDCMotor(inertia=2, torque=1, kv=10, rv=0, fv=0)
    motor._rpm = 0
    motor._pos = 0
    motor._volt = 3
    motor._t = 0

    with patch.object(fake_dc_motor_module, 'time', return_value=2):
      speed = motor.get_speed()

    self.assertAlmostEqual(speed, 29)
    self.assertAlmostEqual(motor._rpm, 29)
    self.assertAlmostEqual(motor._pos, 29 / 60)
    self.assertEqual(motor._t, 2)

  def test_get_position_uses_turns_not_rpm_seconds(self) -> None:
    """Checks that RPM is converted to turns when integrating position."""

    motor = FakeDCMotor(inertia=1, kv=0, rv=-1, fv=0)
    motor._rpm = 60
    motor._pos = 10
    motor._volt = 0
    motor._t = 0

    with patch.object(fake_dc_motor_module, 'time', return_value=1):
      position = motor.get_position()

    self.assertAlmostEqual(position, 11)

  def test_simulation_speed_scales_elapsed_time(self) -> None:
    """Checks the simulation speed multiplier in the state update."""

    motor = FakeDCMotor(inertia=1, kv=0, rv=-1, fv=0, simulation_speed=2)
    motor._rpm = 60
    motor._pos = 0
    motor._volt = 0
    motor._t = 0

    with patch.object(fake_dc_motor_module, 'time', return_value=1):
      motor.get_position()

    self.assertAlmostEqual(motor._pos, 2)
    self.assertEqual(motor._t, 2)

  def test_stop_sets_voltage_to_zero(self) -> None:
    """Checks the inherited stop implementation on the fake motor."""

    motor = FakeDCMotor(inertia=1, kv=0, rv=-1, fv=0)
    motor._rpm = 0
    motor._pos = 0
    motor._volt = 7
    motor._t = 0

    with patch.object(fake_dc_motor_module, 'time', return_value=0):
      motor.stop()

    self.assertEqual(motor._volt, 0)
