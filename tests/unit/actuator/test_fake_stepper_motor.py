# coding: utf-8

import logging
from unittest import TestCase
from unittest.mock import patch
from crappy.actuator.fake_stepper_motor import FakeStepperMotor
import crappy.actuator.fake_stepper_motor as fake_stepper_motor_module


class FakeThread:
  """Small Thread test double for FakeStepperMotor lifecycle tests."""

  def __init__(self, alive: bool = False, alive_after_join: bool = False
               ) -> None:
    """Initializes the fake thread state."""

    self.alive = alive
    self.alive_after_join = alive_after_join
    self.started = False
    self.join_calls = list()

  def start(self) -> None:
    """Records thread start calls."""

    self.started = True
    self.alive = True

  def join(self, timeout: float) -> None:
    """Records join calls and optionally marks the thread as stopped."""

    self.join_calls.append(timeout)
    self.alive = self.alive_after_join

  def is_alive(self) -> bool:
    """Returns whether the fake thread is alive."""

    return self.alive


class TestFakeStepperMotor(TestCase):
  """Unit tests for the FakeStepperMotor Actuator."""

  @staticmethod
  def _capture_logs(motor: FakeStepperMotor) -> list[tuple[int, str]]:
    """Captures log calls made by the fake stepper."""

    logs = list()
    motor.log = lambda level, msg: logs.append((level, msg))
    return logs

  def _run_one_iteration(self,
                         motor: FakeStepperMotor,
                         now: float) -> None:
    """Runs one emulation-thread iteration with a deterministic clock."""

    def stop_after_sleep(_: float) -> None:
      motor._stop_thread = True

    try:
      with (patch.object(fake_stepper_motor_module,
                         'sleep',
                         side_effect=stop_after_sleep),
            patch.object(fake_stepper_motor_module,
                         'monotonic',
                         return_value=now)):
        motor._thread_target()
    finally:
      motor._stop_thread = False

  def test_zero_conversion_and_motion_parameters_are_rejected(self) -> None:
    """Checks validation of parameters that would later divide by zero."""

    cases = (
      {'steps_per_mm': 0},
      {'microsteps': 0},
      {'acceleration': 0},
    )

    for kwargs in cases:
      with self.subTest(kwargs=kwargs):
        with self.assertRaises(ValueError):
          FakeStepperMotor(**kwargs)

  def test_negative_parameters_are_normalized(self) -> None:
    """Parameter signs do not reverse acceleration or the speed limit."""

    for sign in (-1, 1):
      with self.subTest(sign=sign):
        motor = FakeStepperMotor(steps_per_mm=sign * 2, microsteps=sign * 4,
                                 acceleration=sign * 3, max_speed=sign * 5)
        motor._t = 0
        motor.set_speed(100)
        self._run_one_iteration(motor, now=1)
        self.assertEqual(motor.get_speed(), 3)
        self.assertEqual(motor.get_position(), 1.5)
        self._run_one_iteration(motor, now=3)
        self.assertEqual(motor.get_speed(), 5)

  def test_open_starts_emulation_thread(self) -> None:
    """Checks that open starts the worker thread."""

    motor = FakeStepperMotor()
    thread = FakeThread()
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.open()

    self.assertTrue(thread.started)
    self.assertTrue(any(level == logging.INFO for level, _ in logs))

  def test_close_stops_alive_emulation_thread(self) -> None:
    """Checks graceful thread shutdown."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=True)
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.close()

    self.assertFalse(thread.is_alive())
    self.assertTrue(thread.join_calls)
    self.assertTrue(any(level == logging.INFO for level, _ in logs))

  def test_close_logs_error_when_thread_survives_join(self) -> None:
    """Checks the warning path for a stuck emulation thread."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=True, alive_after_join=True)
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.close()

    self.assertTrue(thread.is_alive())
    self.assertTrue(any(level >= logging.ERROR for level, _ in logs))

  def test_close_ignores_thread_that_was_never_started(self) -> None:
    """Checks that close is harmless before open."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=False)
    motor._thread = thread
    motor._stop_thread = False

    motor.close()

    self.assertEqual(thread.join_calls, [])

  def test_set_position_sets_target_and_optional_positive_speed(self) -> None:
    """Position commands replace speed mode and honor an optional magnitude."""

    motor = FakeStepperMotor(steps_per_mm=10, microsteps=2, max_speed=3)
    motor._t = 0
    motor.set_speed(2)
    motor.set_position(100, None)
    self._run_one_iteration(motor, now=1)
    self.assertEqual(motor.get_speed(), 3)
    self.assertGreater(motor.get_position(), 0)

    position = motor.get_position()
    motor.set_position(-100, -4)
    self._run_one_iteration(motor, now=2)
    self.assertEqual(motor.get_speed(), -4)
    self.assertLess(motor.get_position(), position)

  def test_set_speed_sets_target_and_clamps_to_max_speed(self) -> None:
    """Speed commands move in user units and clip excess magnitude."""

    motor = FakeStepperMotor(steps_per_mm=10, microsteps=2, max_speed=3)
    motor._t = 0
    motor.set_speed(2)
    self._run_one_iteration(motor, now=1)
    self.assertEqual(motor.get_speed(), 2)

    logs = self._capture_logs(motor)
    motor.set_speed(-5)
    self._run_one_iteration(motor, now=2)
    self.assertEqual(motor.get_speed(), -3)
    self.assertTrue(any(level >= logging.WARNING for level, _ in logs))

  def test_motion_is_reported_in_user_units(self) -> None:
    """A command reports speed in mm/s and traveled distance in mm."""

    motor = FakeStepperMotor(steps_per_mm=10, microsteps=2, acceleration=10)
    motor._t = 0
    motor.set_speed(2)
    self._run_one_iteration(motor, now=1)
    self.assertEqual(motor.get_speed(), 2)
    self.assertAlmostEqual(motor.get_position(), 1.8)

  def test_stop_zeros_speed_and_switches_to_zero_speed_mode(self) -> None:
    """Stopping cancels a pending position command and leaves the motor still."""

    motor = FakeStepperMotor()
    motor._t = 0
    motor.set_speed(5)
    self._run_one_iteration(motor, now=1)
    self.assertEqual(motor.get_speed(), 5)
    position = motor.get_position()
    motor.set_position(10, None)
    motor.stop()
    self.assertEqual(motor.get_speed(), 0)
    self._run_one_iteration(motor, now=2)
    self.assertEqual(motor.get_position(), position)
    self.assertEqual(motor.get_speed(), 0)

  def test_thread_iteration_accelerates_towards_speed_target(self) -> None:
    """Checks speed-mode acceleration before reaching the target."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=10,
                             max_speed=100)
    motor._t = 0
    motor._target_pos = None
    motor._target_speed = 50

    self._run_one_iteration(motor, now=2)

    self.assertEqual(motor.get_speed(), 20)
    self.assertEqual(motor.get_position(), 20)
    self._run_one_iteration(motor, now=3)
    self.assertEqual(motor.get_speed(), 30)
    self.assertEqual(motor.get_position(), 45)

  def test_thread_iteration_lands_on_speed_target(self) -> None:
    """Checks speed-mode acceleration and deceleration target crossing."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=10,
                             max_speed=100)
    motor._t = 0
    motor._target_pos = None
    motor._target_speed = 5

    self._run_one_iteration(motor, now=1)

    self.assertEqual(motor.get_speed(), 5)
    self.assertEqual(motor.get_position(), 3)

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=10,
                             max_speed=100)
    motor._t = 0
    motor._speed = 5
    motor._target_pos = None
    motor._target_speed = 0

    self._run_one_iteration(motor, now=1)

    self.assertEqual(motor.get_speed(), 0)
    self.assertEqual(motor.get_position(), 1)

  def test_thread_iteration_clamps_above_positive_max_speed(self) -> None:
    """Checks the positive overspeed correction branch."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=100,
                             max_speed=10)
    motor._t = 0
    motor._speed = 12
    motor._target_pos = None
    motor._target_speed = None

    self._run_one_iteration(motor, now=0.05)

    self.assertEqual(motor.get_speed(), 10)

  def test_thread_iteration_clamps_below_negative_max_speed(self) -> None:
    """Checks the negative overspeed correction branch keeps its sign."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=100,
                             max_speed=10)
    motor._t = 0
    motor._speed = -12
    motor._target_pos = None
    motor._target_speed = None

    self._run_one_iteration(motor, now=0.05)

    self.assertEqual(motor.get_speed(), -10)

  def test_thread_iteration_moves_towards_position_targets(self) -> None:
    """Checks position-mode acceleration in both directions."""

    cases = (
      (10000, 10, 5),
      (-10000, -10, -5),
    )

    for target, expected_speed, expected_pos in cases:
      with self.subTest(target=target):
        motor = FakeStepperMotor(steps_per_mm=1,
                                 microsteps=1,
                                 acceleration=10,
                                 max_speed=100)
        motor._t = 0
        motor._target_pos = target
        motor._target_speed = None

        self._run_one_iteration(motor, now=1)

        self.assertEqual(motor.get_speed(), expected_speed)
        self.assertEqual(motor.get_position(), expected_pos)

  def test_thread_iteration_reaches_close_position_target(self) -> None:
    """Checks that a close target can be reached exactly."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=100,
                             max_speed=100)
    motor._t = 0
    motor._speed = 10
    motor._target_pos = 5
    motor._target_speed = None

    self._run_one_iteration(motor, now=1)

    self.assertEqual(motor.get_speed(), 0)
    self.assertEqual(motor.get_position(), 5)

  def test_thread_iteration_leaves_reached_position_target_unchanged(
      self) -> None:
    """Checks the idle-at-target position branch."""

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=100,
                             max_speed=100)
    motor._t = 0
    motor._pos = 7
    motor._speed = 0
    motor._target_pos = 7
    motor._target_speed = None

    self._run_one_iteration(motor, now=1)

    self.assertEqual(motor.get_position(), 7)
    self.assertEqual(motor.get_speed(), 0)
    self._run_one_iteration(motor, now=2)
    self.assertEqual(motor.get_position(), 7)
