"""DC motor simulation observed through commands, speed, and position."""

from unittest import TestCase
from unittest.mock import patch

from crappy.actuator.fake_dc_motor import FakeDCMotor
import crappy.actuator.fake_dc_motor as motor_module


class TestFakeDCMotor(TestCase):
  def test_inertia_cannot_be_zero(self) -> None:
    with self.assertRaises((TypeError, ValueError)):
      FakeDCMotor(inertia=0)

  def test_negative_inertia_is_normalized(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motors = [FakeDCMotor(inertia=inertia, kv=1, rv=-1, fv=0)
                for inertia in (-2, 2)]
      for motor in motors:
        motor.open()
        motor.set_speed(3)
      clock.return_value = 2
      for motor in motors:
        with self.subTest(motor=motor):
          self.assertAlmostEqual(motor.get_speed(), 3)
          self.assertAlmostEqual(motor.get_position(), 0.05)

  def test_open_initializes_state_from_constructor_arguments(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(initial_speed=12, initial_pos=3,
                          simulation_speed=4, kv=1, rv=-1, fv=0)
      motor.open()
      motor.set_speed(5)
      clock.return_value = 1
      self.assertGreater(motor.get_speed(), 12)
      self.assertGreater(motor.get_position(), 3)

      motor.open()
      self.assertEqual(motor.get_speed(), 12)
      self.assertEqual(motor.get_position(), 3)
      clock.return_value = 2
      self.assertEqual(motor.get_speed(), 12)
      self.assertAlmostEqual(motor.get_position(), 3.8)

  def test_voltage_command_applies_only_to_future_motion(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(inertia=1, kv=1, rv=-1, fv=0, initial_speed=60)
      motor.open()
      clock.return_value = 1
      motor.set_speed(5)
      self.assertEqual(motor.get_speed(), 60)
      self.assertEqual(motor.get_position(), 1)
      clock.return_value = 2
      self.assertEqual(motor.get_speed(), 65)

  def test_get_speed_updates_rpm_and_position(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(inertia=2, torque=1, kv=10, rv=0, fv=0)
      motor.open()
      motor.set_speed(3)
      clock.return_value = 2
      self.assertAlmostEqual(motor.get_speed(), 29)
      self.assertAlmostEqual(motor.get_position(), 29 / 60)

  def test_get_position_uses_turns_not_rpm_seconds(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(inertia=1, kv=0, rv=-1, fv=0,
                          initial_speed=60, initial_pos=10)
      motor.open()
      clock.return_value = 1
      self.assertAlmostEqual(motor.get_position(), 11)

  def test_simulation_speed_scales_elapsed_time(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(kv=0, rv=-1, fv=0, initial_speed=60,
                          simulation_speed=2)
      motor.open()
      clock.return_value = 1
      self.assertAlmostEqual(motor.get_position(), 2)

  def test_stop_sets_voltage_to_zero(self) -> None:
    with patch.object(motor_module, 'monotonic', return_value=0) as clock:
      motor = FakeDCMotor(inertia=1, kv=1, rv=-1, fv=0)
      motor.open()
      motor.set_speed(7)
      clock.return_value = 1
      self.assertEqual(motor.get_speed(), 7)
      motor.stop()
      clock.return_value = 2
      self.assertEqual(motor.get_speed(), 7)
