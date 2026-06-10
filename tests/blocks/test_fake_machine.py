# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch
from crappy.blocks.fake_machine import FakeMachine
import crappy.blocks.fake_machine as fake_machine_module

from ..block import BlockTestBase


class TestFakeMachine(BlockTestBase):
  """Unit tests for the FakeMachine Block-specific behavior."""

  _t0 = 10.0

  def _make_machine(self, **kwargs) -> FakeMachine:
    """Creates a deterministic FakeMachine ready for direct method calls."""

    kwargs.setdefault('sigma', dict())
    machine = FakeMachine(**kwargs)
    machine._instance_t0 = Value('d', self._t0)
    return machine

  @staticmethod
  def _capture_send(machine: FakeMachine) -> list[dict[str, Any]]:
    """Captures values sent by a FakeMachine without going through Links."""

    sent = list()

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    machine.send = send
    return sent

  @staticmethod
  def _set_received(machine: FakeMachine,
                    data: dict[str, Any]) -> list[bool]:
    """Makes recv_last_data return the requested input command."""

    fill_missing_values = list()

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      fill_missing_values.append(fill_missing)
      return dict(data)

    machine.recv_last_data = recv_last_data
    return fill_missing_values

  def _assert_sent_values(self,
                          values: dict[str, Any], *,
                          t: float,
                          force: float,
                          position: float,
                          exx: float,
                          eyy: float) -> None:
    """Checks the complete payload emitted by FakeMachine."""

    self.assertEqual(set(values), {'t(s)', 'F(N)', 'x(mm)',
                                   'Exx(%)', 'Eyy(%)'})
    self.assertAlmostEqual(values['t(s)'], t)
    self.assertAlmostEqual(values['F(N)'], force)
    self.assertAlmostEqual(values['x(mm)'], position)
    self.assertAlmostEqual(values['Exx(%)'], exx)
    self.assertAlmostEqual(values['Eyy(%)'], eyy)

  def test_begin_sends_initial_state(self) -> None:
    """Checks that begin initializes timing and emits the zero state."""

    machine = self._make_machine(rigidity=1000, l0=100, nu=0.25)
    sent = self._capture_send(machine)

    with patch.object(fake_machine_module, 'time', return_value=10.25):
      machine.begin()

    self.assertEqual(machine._prev_t, self._t0)
    self.assertEqual(len(sent), 1)
    self._assert_sent_values(sent[0],
                             t=0.25,
                             force=0,
                             position=0,
                             exx=0,
                             eyy=0)

  def test_loop_returns_if_command_label_is_missing(self) -> None:
    """Checks that missing commands leave the machine state untouched."""

    machine = self._make_machine(cmd_label='drive')
    machine._prev_t = self._t0
    machine._current_pos = 1.5
    sent = self._capture_send(machine)
    fill_missing_values = self._set_received(machine, {'cmd': 5})

    machine.loop()

    self.assertEqual(fill_missing_values, [True])
    self.assertEqual(machine._prev_t, self._t0)
    self.assertEqual(machine._current_pos, 1.5)
    self.assertEqual(sent, [])

  def test_speed_mode_updates_position_and_outputs(self) -> None:
    """Checks speed mode, including max_speed clamping and reverse motion."""

    cases = (
      {'cmd': 2, 'current_pos': 0, 'delta_t': 0.5, 'position': 1},
      {'cmd': 20, 'current_pos': 0, 'delta_t': 2, 'position': 10},
      {'cmd': -20, 'current_pos': 20, 'delta_t': 1, 'position': 15},
    )

    for case in cases:
      with self.subTest(case=case):
        machine = self._make_machine(rigidity=200,
                                     l0=100,
                                     max_strain=100,
                                     max_speed=5,
                                     plastic_law=lambda _: 0,
                                     mode='speed')
        machine._prev_t = self._t0
        machine._current_pos = case['current_pos']
        sent = self._capture_send(machine)
        fill_missing_values = self._set_received(machine, {'cmd': case['cmd']})

        t = self._t0 + case['delta_t']
        with patch.object(fake_machine_module, 'time', side_effect=(t, t)):
          machine.loop()

        position = case['position']
        self.assertEqual(fill_missing_values, [True])
        self.assertEqual(len(sent), 1)
        self._assert_sent_values(sent[0],
                                 t=case['delta_t'],
                                 force=position * 2,
                                 position=position,
                                 exx=position,
                                 eyy=-0.3 * position)

  def test_position_mode_reaches_or_clamps_to_target(self) -> None:
    """Checks position mode for reachable, clamped, and reverse moves."""

    cases = (
      {'cmd': 3, 'current_pos': 0, 'delta_t': 2, 'position': 3},
      {'cmd': 20, 'current_pos': 0, 'delta_t': 2, 'position': 10},
      {'cmd': 4, 'current_pos': 10, 'delta_t': 3, 'position': 4},
    )

    for case in cases:
      with self.subTest(case=case):
        machine = self._make_machine(rigidity=200,
                                     l0=100,
                                     max_strain=100,
                                     max_speed=5,
                                     plastic_law=lambda _: 0,
                                     mode='position',
                                     cmd_label='target')
        machine._prev_t = self._t0
        machine._current_pos = case['current_pos']
        sent = self._capture_send(machine)
        fill_missing_values = self._set_received(
          machine, {'target': case['cmd']})

        t = self._t0 + case['delta_t']
        with patch.object(fake_machine_module, 'time', side_effect=(t, t)):
          machine.loop()

        position = case['position']
        self.assertEqual(fill_missing_values, [True])
        self.assertEqual(len(sent), 1)
        self._assert_sent_values(sent[0],
                                 t=case['delta_t'],
                                 force=position * 2,
                                 position=position,
                                 exx=position,
                                 eyy=-0.3 * position)

  def test_plastic_law_reduces_elastic_force(self) -> None:
    """Checks that the custom plastic law contributes to the emitted force."""

    def plastic_law(strain: float) -> float:
      return strain / 2

    machine = self._make_machine(rigidity=200,
                                 l0=100,
                                 max_strain=100,
                                 max_speed=5,
                                 plastic_law=plastic_law)
    machine._prev_t = self._t0
    sent = self._capture_send(machine)
    self._set_received(machine, {'cmd': 20})

    with patch.object(fake_machine_module, 'time', side_effect=(12.0, 12.0)):
      machine.loop()

    self.assertEqual(machine._max_recorded_strain, 0.1)
    self.assertEqual(machine._plastic_elongation, 5)
    self._assert_sent_values(sent[0],
                             t=2,
                             force=10,
                             position=10,
                             exx=10,
                             eyy=-3)

  def test_sample_break_sets_force_to_zero(self) -> None:
    """Checks that exceeding max_strain cancels rigidity before sending."""

    machine = self._make_machine(rigidity=200,
                                 l0=100,
                                 max_strain=5,
                                 max_speed=5)
    machine._prev_t = self._t0
    machine._prev_broke_t = 11.5
    sent = self._capture_send(machine)
    self._set_received(machine, {'cmd': 20})

    with patch.object(fake_machine_module, 'time',
                      side_effect=(12.0, 12.0, 12.0)):
      machine.loop()

    self.assertEqual(machine._rigidity, 0)
    self._assert_sent_values(sent[0],
                             t=2,
                             force=0,
                             position=10,
                             exx=10,
                             eyy=-3)

  def test_add_noise_applies_requested_sigmas_only(self) -> None:
    """Checks that sigma controls which labels receive Gaussian noise."""

    machine = self._make_machine(sigma={'F(N)': 2, 'x(mm)': 0.5})
    to_send = {'t(s)': 1, 'F(N)': 10, 'x(mm)': 3,
               'Exx(%)': 4, 'Eyy(%)': -1}

    calls = list()

    def normal(value: float, sigma: float) -> float:
      calls.append((value, sigma))
      return value + 10 * sigma

    with patch.object(fake_machine_module.np.random, 'normal',
                      side_effect=normal):
      noised = machine._add_noise(to_send)

    self.assertIs(noised, to_send)
    self.assertEqual(calls, [(10, 2), (3, 0.5)])
    self.assertEqual(noised, {'t(s)': 1, 'F(N)': 30, 'x(mm)': 8,
                              'Exx(%)': 4, 'Eyy(%)': -1})

  def test_invalid_mode_raises(self) -> None:
    """Checks the runtime validation of the mode argument."""

    with self.assertRaises(ValueError):
      self._make_machine(mode='invalid')
