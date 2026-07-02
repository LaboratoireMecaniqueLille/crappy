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
                         'time',
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
    """Checks that magnitudes are stored as positive internal values."""

    motor = FakeStepperMotor(steps_per_mm=-2,
                             microsteps=-4,
                             acceleration=-3,
                             max_speed=-5)

    self.assertEqual(motor._steps_per_mm, 2)
    self.assertEqual(motor._microsteps, 4)
    self.assertEqual(motor._accel, 24)
    self.assertEqual(motor._max_speed, 40)

  def test_open_starts_emulation_thread(self) -> None:
    """Checks that open starts the worker thread."""

    motor = FakeStepperMotor()
    thread = FakeThread()
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.open()

    self.assertTrue(thread.started)
    self.assertEqual(logs, [(logging.INFO, "Starting the emulation thread")])

  def test_close_stops_alive_emulation_thread(self) -> None:
    """Checks graceful thread shutdown."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=True)
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.close()

    self.assertTrue(motor._stop_thread)
    self.assertEqual(thread.join_calls, [0.1])
    self.assertEqual(logs, [(logging.INFO,
                             "Trying to stop the emulation thread")])

  def test_close_logs_error_when_thread_survives_join(self) -> None:
    """Checks the warning path for a stuck emulation thread."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=True, alive_after_join=True)
    motor._thread = thread
    logs = self._capture_logs(motor)

    motor.close()

    self.assertIn((logging.ERROR,
                   "The emulation thread did not terminate properly !"),
                  logs)

  def test_close_ignores_thread_that_was_never_started(self) -> None:
    """Checks that close is harmless before open."""

    motor = FakeStepperMotor()
    thread = FakeThread(alive=False)
    motor._thread = thread
    motor._stop_thread = False

    motor.close()

    self.assertFalse(motor._stop_thread)
    self.assertEqual(thread.join_calls, [])

  def test_set_position_sets_target_and_optional_positive_speed(self) -> None:
    """Checks position command conversion and speed magnitude handling."""

    motor = FakeStepperMotor(steps_per_mm=10,
                             microsteps=2,
                             max_speed=3)
    motor._target_speed = 12

    motor.set_position(1.5, None)

    self.assertEqual(motor._target_pos, 30)
    self.assertIsNone(motor._target_speed)
    self.assertEqual(motor._max_speed, 60)

    motor.set_position(-2, -4)

    self.assertEqual(motor._target_pos, -40)
    self.assertEqual(motor._max_speed, 80)

  def test_set_speed_sets_target_and_clamps_to_max_speed(self) -> None:
    """Checks speed command conversion and clipping."""

    motor = FakeStepperMotor(steps_per_mm=10,
                             microsteps=2,
                             max_speed=3)
    logs = self._capture_logs(motor)

    motor.set_speed(2)

    self.assertIsNone(motor._target_pos)
    self.assertEqual(motor._target_speed, 40)
    self.assertEqual(logs, [])

    motor.set_speed(-5)

    self.assertEqual(motor._target_speed, -60)
    self.assertEqual(logs[-1][0], logging.WARNING)

  def test_getters_convert_internal_steps_to_user_units(self) -> None:
    """Checks speed and position conversions back to mm and mm/s."""

    motor = FakeStepperMotor(steps_per_mm=10, microsteps=2)
    motor._speed = 40
    motor._pos = -20

    self.assertEqual(motor.get_speed(), 2)
    self.assertEqual(motor.get_position(), -1)

  def test_stop_zeros_speed_and_switches_to_zero_speed_mode(self) -> None:
    """Checks the abrupt stop behavior."""

    motor = FakeStepperMotor()
    logs = self._capture_logs(motor)
    motor._speed = 5
    motor._target_pos = 10
    motor._target_speed = 2

    motor.stop()

    self.assertEqual(motor._speed, 0)
    self.assertIsNone(motor._target_pos)
    self.assertEqual(motor._target_speed, 0)
    self.assertEqual(logs, [(logging.INFO,
                             "Abruptly stopping the emulated stepper motor")])

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

    self.assertEqual(motor._speed, 20)
    self.assertEqual(motor._pos, 20)
    self.assertEqual(motor._t, 2)

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

    self.assertEqual(motor._speed, 5)
    self.assertEqual(motor._pos, 3)

    motor = FakeStepperMotor(steps_per_mm=1,
                             microsteps=1,
                             acceleration=10,
                             max_speed=100)
    motor._t = 0
    motor._speed = 5
    motor._target_pos = None
    motor._target_speed = 0

    self._run_one_iteration(motor, now=1)

    self.assertEqual(motor._speed, 0)
    self.assertEqual(motor._pos, 1)

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

    self.assertEqual(motor._speed, 10)

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

    self.assertEqual(motor._speed, -10)

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

        self.assertEqual(motor._speed, expected_speed)
        self.assertEqual(motor._pos, expected_pos)

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

    self.assertEqual(motor._speed, 0)
    self.assertEqual(motor._pos, 5)

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

    self.assertEqual(motor._pos, 7)
    self.assertEqual(motor._speed, 0)
    self.assertEqual(motor._t, 1)
