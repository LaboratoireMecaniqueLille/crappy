# coding: utf-8

import crappy
import unittest
import numpy as np
from typing import List, Dict


class FakeInOutOffsetList(crappy.InOut):
  """"""

  def get_data(self) -> List[float]:
    """"""

    return[-1, 0, 1, 2]

  def start_stream(self) -> None:
    """"""

    ...

  def get_stream(self) -> List[np.ndarray]:
    """"""

    t = -1
    t_data = np.array([t + 0, t + 1, t + 2, t + 3])
    data = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    return [t_data, data]

  def stop_stream(self) -> None:
    """"""

    ...


class FakeInOutOffsetDict(crappy.InOut):
  """"""

  def get_data(self) -> Dict[str, float]:
    """"""

    return {'t(s)': -1, 'a': 0, 'b': 1, 'c': 2}

  def start_stream(self) -> None:
    """"""

    ...

  def get_stream(self) -> Dict[str, np.ndarray]:
    """"""

    t = -1
    t_data = np.array([t + 0, t + 1, t + 2, t + 3])
    data = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    return {'t(s)': t_data, 'stream': data}

  def stop_stream(self) -> None:
    """"""

    ...


class TestInOutOffset(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    crappy.reset()

  def test_offset_list(self) -> None:
    """"""

    # Initialization
    inout = FakeInOutOffsetList()
    inout.make_zero(2)

    # Testing get_data
    self.assertIsInstance(inout.get_data(), list)
    self.assertListEqual(inout.get_data(), [-1, 0, 1, 2])

    # Testing return_data
    self.assertIsInstance(inout.return_data(), list)
    self.assertListEqual(inout.return_data(), [-1, 0, 0, 0])

    # Testing get_stream
    self.assertIsInstance(inout.get_stream(), list)
    self.assertTrue((inout.get_stream()[0] == np.array([-1, 0, 1, 2])).all())
    self.assertTrue((inout.get_stream()[1] ==
                     np.array([[0, 1, 2], [0, 1, 2],
                               [0, 1, 2], [0, 1, 2]])).all())

    # Testing return_stream
    self.assertIsInstance(inout.return_stream(), list)
    self.assertTrue((inout.return_stream()[0] ==
                     np.array([-1, 0, 1, 2])).all())
    self.assertTrue((inout.return_stream()[1] ==
                     np.array([[0, 0, 0], [0, 0, 0],
                               [0, 0, 0], [0, 0, 0]])).all())

  def test_offset_dict(self) -> None:
    """"""

    # Initialization
    inout = FakeInOutOffsetDict()
    inout.make_zero(2)

    # Testing get_data
    self.assertIsInstance(inout.get_data(), dict)
    self.assertDictEqual(inout.get_data(),
                         {'t(s)': -1, 'a': 0, 'b': 1, 'c': 2})

    # Testing return_data
    self.assertIsInstance(inout.return_data(), dict)
    self.assertDictEqual(inout.return_data(),
                         {'t(s)': -1, 'a': 0, 'b': 0, 'c': 0})

    # Testing get_stream
    self.assertIsInstance(inout.get_stream(), dict)
    self.assertTrue((inout.get_stream()['t(s)'] ==
                     np.array([-1, 0, 1, 2])).all())
    self.assertTrue((inout.get_stream()['stream'] ==
                     np.array([[0, 1, 2], [0, 1, 2],
                               [0, 1, 2], [0, 1, 2]])).all())

    # Testing return_stream
    self.assertIsInstance(inout.return_stream(), dict)
    self.assertTrue((inout.return_stream()['t(s)'] ==
                     np.array([-1, 0, 1, 2])).all())
    self.assertTrue((inout.return_stream()['stream'] ==
                     np.array([[0, 0, 0], [0, 0, 0],
                               [0, 0, 0], [0, 0, 0]])).all())
