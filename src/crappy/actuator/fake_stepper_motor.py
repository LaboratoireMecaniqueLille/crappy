# coding: utf-8

from typing import Optional
from time import time, sleep
from math import sqrt, copysign
from threading import Thread, RLock
import logging

from .meta_actuator import Actuator


class FakeStepperMotor(Actuator):
  """This :class:`~crappy.actuator.Actuator` can emulate the
  behavior of a stepper motor used as a linear actuator.

  It can drive the motor either in speed or in position, unlike the other fake
  Actuator :class:`~crappy.actuator.FakeDCMotor` that can only be driven in
  speed. This class can also return the current speed and position of the
  motor.

  Internally, the behavior of the motor is emulated in a separate
  :obj:`~threading.Thread` and is based on the fundamental equations of the
  constantly accelerated linear movement.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self,
               steps_per_mm: float = 100,
               microsteps: int = 256,
               acceleration: float = 20,
               max_speed: float = 10) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      steps_per_mm: The number of full steps needed for the motor to move by 1
        mm. The higher this value, the more precise the motor is but the lower
        the maximum achievable speed.
      microsteps: The number of microsteps into which a full step is divided.
        The higher this value, the more precise the resolution of the motor,
        but the lower the maximum achievable speed.
      acceleration: The maximum acceleration the motor can achieve, in
        mm/s². In this demo Actuator, this parameter only has an incidence
        on the responsiveness of the motor.
      max_speed: The maximum achievable speed of the motor, in mm/s.
    """

    super().__init__()

    # The variables describing the motor
    self._accel: float = abs(acceleration * steps_per_mm * microsteps)
    self._max_speed: float = abs(max_speed * steps_per_mm * microsteps)
    self._steps_per_mm: float = steps_per_mm
    self._microsteps: int = microsteps

    # The variables for driving the motor
    self._speed: float = 0
    self._pos: float = 0
    self._target_pos: Optional[float] = 0
    self._target_speed: Optional[float] = None
    self._t: float = time()

    # Attributes managing the Thread emulating the motor
    self._stop_thread: bool = False
    self._lock = RLock()
    self._thread = Thread(target=self._thread_target)

  def open(self) -> None:
    """Starts the :obj:`~threading.Thread` emulating the stepper motor."""

    self.log(logging.INFO, "Starting the emulation thread")
    self._thread.start()

  def set_position(self, position: float, speed: Optional[float]) -> None:
    """Sets the target position that the motor should reach.

    It first converts the given position from mm to steps. If a speed value is
    given, also sets the maximum speed to the given value.

    Args:
      position: The target position to reach, in mm.
      speed: Optionally, the maximum speed at which to move for moving to the
        target position, in mm/s. If the current and target position are too
        close, the motor might not reach this position during its movement.
    """

    with self._lock:
      self._target_speed = None
      self._target_pos = position * self._steps_per_mm * self._microsteps
      if speed is not None:
        self._max_speed = speed * self._steps_per_mm * self._microsteps
      self.log(logging.DEBUG, f"Set the target position to {self._target_pos}"
                              f", with speed {self._max_speed}")

  def set_speed(self, speed: float) -> None:
    """Sets the target speed that the motor should reach.

    It first converts the speed to steps/s, and also checks that the command is
    consistent with the given maximum speed.

    Args:
      speed: The target speed of the motor, in mm/s. Can be negative.
    """

    with self._lock:
      self._target_pos = None
      speed = speed * self._steps_per_mm * self._microsteps
      if abs(speed) > self._max_speed:
        speed = copysign(self._max_speed, speed)
        self.log(logging.WARNING,
                 f"Target speed {speed} higher than the maximum allowed speed "
                 f"{self._max_speed}, setting to max speed")
      self._target_speed = speed

  def get_speed(self) -> float:
    """Returns the current speed of the motor, in mm/s."""

    with self._lock:
      return self._speed / self._steps_per_mm / self._microsteps

  def get_position(self) -> float:
    """Returns the current position of the motor, in mm."""

    with self._lock:
      return self._pos / self._steps_per_mm / self._microsteps

  def stop(self) -> None:
    """Instantly sets the speed and the speed command to 0."""

    with self._lock:
      self.log(logging.INFO, "Abruptly stopping the emulated stepper motor")
      self._speed = 0
      self._target_pos = None
      self._target_speed = 0

  def close(self) -> None:
    """Stops the :obj:`~threading.Thread` emulating the stepper motor if it was
    started."""

    if self._thread.is_alive():
      self.log(logging.INFO, "Trying to stop the emulation thread")
      self._stop_thread = True
      self._thread.join(0.1)
      if self._thread.is_alive():
        self.log(logging.ERROR, "The emulation thread did not terminate "
                                "properly !")

  def _thread_target(self) -> None:
    """The target for the Thread emulating the stepper motor behavior.

    It updates the speed and position of the motor so that the motor reaches
    the target speed or position in the fastest possible way. It is mostly a
    decision tree managing the many possible cases.
    """

    # Emulating the behavior of the motor until told to stop
    while not self._stop_thread:

      # To avoid spamming the CPU
      sleep(0.001)

      # Using a Lock to avoid race conditions with the main Thread
      with self._lock:

        # Constant that will be useful later
        c1 = self._max_speed ** 2 / self._accel

        # Updating the last timestamp
        t = time()
        delta_t = t - self._t
        self._t = t

        # If above the max speed, ignore the target and decelerate
        if self._speed > self._max_speed:
          # Case when we reach the max speed during delta_t
          if self._speed - self._accel * delta_t < self._max_speed:
            self._pos += int(self._max_speed * delta_t
                             + 0.5 * (self._speed - self._max_speed) ** 2
                             / self._accel)
            self._speed = self._max_speed
            continue

          # Case when we're still not reaching the max speed
          self._pos += int(-self._accel * delta_t ** 2 / 2
                           + self._speed * delta_t)
          self._speed += -self._accel * delta_t
          continue

        # If below the min speed, ignore the target and accelerate
        if self._speed < -self._max_speed:
          # Case when we reach the min speed during delta_t
          if self._speed + self._accel * delta_t > -self._max_speed:
            self._pos += int(-self._max_speed * delta_t
                             - 0.5 * (self._speed + self._max_speed) ** 2
                             / self._accel)
            self._speed = self._max_speed
            continue

          # Case when we're still not reaching the min speed
          self._pos += int(self._accel * delta_t ** 2 / 2
                           + self._speed * delta_t)
          self._speed += self._accel * delta_t
          continue

        # Driving the motor in position mode
        if self._target_pos is not None:

          # Calculating the difference and a useful constant
          diff = self._target_pos - self._pos
          c2 = self._speed ** 2 / 2 / self._accel

          # Nothing to do if we're already stopped at the target position
          if diff == 0 and self._speed == 0:
            continue

          # Far enough from target to have time to reach the max or min speed
          if abs(diff) >= c1 - c2:
            # Accelerating to the max speed then decelerating to 0
            if diff > 0:

              # We first have to reach the max speed
              if self._speed < self._max_speed:
                # Case when we reach the max speed during delta_t
                if self._speed + self._accel * delta_t > self._max_speed:
                  self._pos += int(self._max_speed * delta_t
                                   - 0.5 * (self._max_speed - self._speed) ** 2
                                   / self._accel)
                  self._speed = self._max_speed
                  continue

                # Case when we're still not reaching the max speed
                self._pos += int(self._accel * delta_t ** 2 / 2
                                 + self._speed * delta_t)
                self._speed += self._accel * delta_t
                continue

              # We need to decelerate at a precise moment
              else:
                # Case when we reach the limit position during delta_t
                if (self._pos + int(self._max_speed * delta_t) >
                    self._target_pos - c1 / 2):
                  self._speed += (
                      -self._accel * (delta_t - diff / self._max_speed +
                                      self._max_speed / 2 / self._accel))
                  self._pos += int(
                      self._max_speed * delta_t - 0.5 * self._accel *
                      (delta_t - diff / self._max_speed + self._max_speed / 2
                       / self._accel) ** 2)
                  continue

                # Case when we're still not reaching the limit position
                self._pos += int(self._max_speed * delta_t)
                self._speed = self._max_speed
                continue

            # Decelerating to the max speed then accelerating to 0
            else:
              # We first have to reach the min speed
              if self._speed > -self._max_speed:
                # Case when we reach the min speed during delta_t
                if self._speed - self._accel * delta_t < -self._max_speed:
                  self._pos += int(-self._max_speed * delta_t
                                   + 0.5 * (self._max_speed + self._speed) ** 2
                                   / self._accel)
                  self._speed = -self._max_speed
                  continue

                # Case when we're still not reaching the min speed
                self._pos += int(-self._accel * delta_t ** 2 / 2
                                 + self._speed * delta_t)
                self._speed += -self._accel * delta_t
                continue

              # We need to accelerate at a precise moment
              else:
                # Case when we reach the limit position during delta_t
                if (self._pos + int(-self._max_speed * delta_t) <
                    self._target_pos + c1 / 2):
                  self._speed += (
                      self._accel * (delta_t + diff / self._max_speed +
                                     self._max_speed / 2 / self._accel))
                  self._pos += int(
                      -self._max_speed * delta_t + 0.5 * self._accel *
                      (delta_t + diff / self._max_speed + self._max_speed / 2
                       / self._accel) ** 2)
                  continue

                # Case when we're still not reaching the limit position
                self._pos += int(-self._max_speed * delta_t)
                self._speed = -self._max_speed
                continue

          # Too close to target to have time to reach max or min speed
          else:
            # The criteria differ if the speed is positive or negative
            if self._speed >= 0:

              # First accelerate then decelerate to reach target
              if diff >= c2:

                # Close enough to the target, able to reach it within delta_t
                if diff / delta_t <= self._speed <= self._accel * delta_t:
                  self._speed = 0
                  self._pos = self._target_pos
                  continue

                v_lim = sqrt(self._speed ** 2 / 2 + self._accel * diff)

                # Case when we reach the limit speed during delta_t
                if self._speed + self._accel * delta_t > v_lim:
                  self._pos += int(
                      v_lim * delta_t - 0.5 * (v_lim - self._speed) ** 2 /
                      self._accel - self._accel / 2 *
                      (delta_t - (v_lim - self._speed) / self._accel) ** 2)
                  self._speed += (2 * (v_lim - self._speed)
                                  - self._accel * delta_t)
                  continue

                # Case when we're still not reaching the limit speed
                self._pos += int(0.5 * self._accel * delta_t ** 2
                                 + self._speed * delta_t)
                self._speed += self._accel * delta_t
                continue

              # First decelerate then accelerate to reach target
              else:

                v_lim = - sqrt(self._speed ** 2 / 2 - self._accel * diff)

                # Case when we reach the limit speed during delta_t
                if self._speed - self._accel * delta_t < v_lim:
                  self._pos += int(
                      v_lim * delta_t + 0.5 * (v_lim - self._speed) ** 2 /
                      self._accel + self._accel / 2 *
                      (delta_t - (self._speed - v_lim) / self._accel) ** 2)
                  self._speed += (2 * (v_lim - self._speed)
                                  + self._accel * delta_t)
                  continue

                # Case when we're still not reaching the limit speed
                self._pos += int(0.5 * -self._accel * delta_t ** 2
                                 + self._speed * delta_t)
                self._speed += -self._accel * delta_t
                continue

            # The criteria differ if the speed is positive or negative
            else:

              # First decelerate then accelerate to reach target
              if -diff >= c2:

                # Close enough to the target, able to reach it within delta_t
                if -self._accel * delta_t <= self._speed <= diff / delta_t:
                  self._speed = 0
                  self._pos = self._target_pos
                  continue

                v_lim = - sqrt(self._speed ** 2 / 2 - self._accel * diff)

                # Case when we reach the limit speed during delta_t
                if self._speed - self._accel * delta_t < v_lim:
                  self._pos += int(
                      v_lim * delta_t + 0.5 * (v_lim - self._speed) ** 2 /
                      self._accel + self._accel / 2 *
                      (delta_t - (self._speed - v_lim) / self._accel) ** 2)
                  self._speed += (2 * (v_lim - self._speed) +
                                  self._accel * delta_t)
                  continue

                # Case when we're still not reaching the limit speed
                self._pos += int(0.5 * -self._accel * delta_t ** 2
                                 + self._speed * delta_t)
                self._speed += -self._accel * delta_t
                continue

              # First accelerate then decelerate to reach target
              else:

                v_lim = sqrt(self._speed ** 2 / 2 + self._accel * diff)

                # Case when we reach the limit speed during delta_t
                if self._speed + self._accel * delta_t > v_lim:
                  self._pos += int(
                      v_lim * delta_t - 0.5 * (v_lim - self._speed) ** 2 /
                      self._accel - self._accel / 2 *
                      (delta_t - (v_lim - self._speed) / self._accel) ** 2)
                  self._speed += (2 * (v_lim - self._speed)
                                  - self._accel * delta_t)
                  continue

                # Case when we're still not reaching the limit speed
                self._pos += int(0.5 * self._accel * delta_t ** 2
                                 + self._speed * delta_t)
                self._speed += self._accel * delta_t
                continue

        # Driving the motor in speed mode
        if self._target_speed is not None:

          diff = self._target_speed - self._speed

          # Just updating the position if we're already at the target speed
          if diff == 0:
            self._pos += int(self._speed * delta_t)
            continue

          # Case when we need to decelerate to reach the target speed
          if diff < 0:
            # Case when we reach the target speed during delta_t
            if self._speed - self._accel * delta_t < self._target_speed:
              self._pos += int((self._target_speed * delta_t
                               + 0.5 * diff ** 2 / self._accel))
              self._speed = self._target_speed
              continue

            # Case when we're still not reaching the target speed
            self._pos += int(-self._accel * delta_t ** 2 / 2
                             + self._speed * delta_t)
            self._speed += -self._accel * delta_t
            continue

          # Case when we need to accelerate to reach the target speed
          else:
            # Case when we reach the target speed during delta_t
            if self._speed + self._accel * delta_t > self._target_speed:
              self._pos += int(self._target_speed * delta_t
                               - 0.5 * diff ** 2 / self._accel)
              self._speed = self._target_speed
              continue

            # Case when we're still not reaching the target speed
            self._pos += int(self._accel * delta_t ** 2 / 2
                             + self._speed * delta_t)
            self._speed += self._accel * delta_t
            continue
