# coding: utf-8

from crappy.actuator import FakeDCMotor
import unittest


class TestFakeMotor(unittest.TestCase):
  """"""

  def setUp(self) -> None:
    """"""

    self._actuator = FakeDCMotor()

  def test_open(self) -> None:
    """"""

    self.assertIsNone(self._actuator.open())

  def test_set_speed(self) -> None:
    """"""

    self._actuator.open()
    self.assertIsNone(self._actuator.set_speed(1))
    self.assertIsNone(self._actuator.set_speed(0))

  def test_get_speed(self) -> None:
    """"""

    self._actuator.open()
    self.assertIsInstance(self._actuator.get_speed(), float)

  def test_get_position(self) -> None:
    """"""

    self._actuator.open()
    self.assertIsInstance(self._actuator.get_position(), float)
