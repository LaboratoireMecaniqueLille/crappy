# coding: utf-8

from crappy.actuator import Actuator
import unittest


class TestActuator(unittest.TestCase):
  """"""

  def setUp(self) -> None:
    """"""

    self._actuator = Actuator()

  def test_open(self) -> None:
    """"""

    self.assertIsNone(self._actuator.open())

  def test_set_speed(self) -> None:
    """"""

    self.assertIsNone(self._actuator.set_speed(0.))

  def test_set_position(self) -> None:
    """"""

    self.assertIsNone(self._actuator.set_position(0., None))

  def test_get_speed(self) -> None:
    """"""

    self.assertIsNone(self._actuator.get_speed())

  def test_get_position(self) -> None:
    """"""

    self.assertIsNone(self._actuator.get_position())

  def test_stop(self) -> None:
    """"""

    self.assertIsNone(self._actuator.stop())

  def test_close(self) -> None:
    """"""

    self.assertIsNone(self._actuator.close())
