# coding: utf-8

import unittest
import crappy
from crappy.links import Link
from crappy.blocks import Machine
from crappy.blocks.machine import ActuatorInstance
from typing import Dict, List, Any
from time import time
from multiprocessing import Value, Queue


class FakeBlock:
  """"""

  def __init__(self) -> None:
    """"""

    self.inputs: List[Link] = list()
    self.outputs: List[Link] = list()

  def send(self, data: Dict[str, Any]) -> None:
    """"""

    for link in self.outputs:
      link.send(data)

  def recv(self) -> Dict[str, Any]:
    """"""

    buf = dict()
    for link in self.inputs:
      buf.update(link.recv_last())

    return buf

  def add_input(self, link: Link)-> None:
    """"""

    self.inputs.append(link)

  def add_output(self, link: Link) -> None:
    """"""

    self.outputs.append(link)


class TestMachine(unittest.TestCase):
  """"""

  def test_machine(self) -> None:
    """"""

    base_dict = dict(actuators=[{'type': 'FakeDCMotor',
                                 'cmd_label': 'cmd',
                                 'mode': 'speed',
                                 'speed': 10,
                                 'position_label': 'pos',
                                 'speed_label': 'speed',
                                 'speed_cmd_label': 'speed_cmd'}],
                     common=None,
                     time_label='t(s)',
                     ft232h_ser_num=None,
                     spam=False,
                     freq=200,
                     display_freq=False,
                     debug=False)

    common_dict = dict(actuators=[{'type': 'FakeDCMotor'}],
                       common={'cmd_label': 'cmd',
                               'mode': 'speed',
                               'speed': 10,
                               'position_label': 'pos',
                               'speed_label': 'speed',
                               'speed_cmd_label': 'speed_cmd'},
                       time_label='t(s)',
                       ft232h_ser_num=None,
                       spam=False,
                       freq=200,
                       display_freq=False,
                       debug=False)

    for dict_ in (base_dict, common_dict):
      with self.subTest(args=dict_):

        fake_input = FakeBlock()
        fake_output = FakeBlock()

        machine = Machine(**dict_)

        t0 = Value('d', time())
        machine._instance_t0 = t0
        log_queue = Queue()
        machine._log_queue = log_queue
        self.assertIsNone(machine._set_block_logger())

        crappy.link(fake_input, machine)
        crappy.link(machine, fake_output)

        self.assertIsNone(machine.prepare())
        self.assertEqual(len(machine._actuators), 1)
        self.assertIsInstance(machine._actuators[0], ActuatorInstance)
        self.assertEqual(machine._actuators[0].speed, 10)

        self.assertIsNone(machine.loop())
        data = fake_output.recv()
        self.assertIn('t(s)', data)
        self.assertIn('pos', data)
        self.assertIn('speed', data)
        self.assertEqual(data['pos'], 0)
        self.assertEqual(data['speed'], 0)

        fake_input.send({'t(s)': time() - t0.value, 'cmd': 100,
                         'speed_cmd': 20})
        fake_input.send({'t(s)': time() - t0.value, 'cmd': 100,
                         'speed_cmd': 20})

        recv = machine.recv_data()
        self.assertIn('t(s)', recv)
        self.assertIn('cmd', recv)
        self.assertIn('speed_cmd', recv)

        self.assertIsNone(machine.loop())
        self.assertEqual(machine._actuators[0].speed, 20)
        data = fake_output.recv()
        self.assertIn('t(s)', data)
        self.assertIn('pos', data)
        self.assertIn('speed', data)
        self.assertGreater(data['pos'], 0)
        self.assertGreater(data['speed'], 0)

        self.assertIsNone(machine.finish())
