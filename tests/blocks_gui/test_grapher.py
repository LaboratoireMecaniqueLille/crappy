# coding: utf-8

from tkinter import TclError
from unittest.mock import patch

from crappy.blocks.grapher import Grapher
import crappy.blocks.grapher as grapher_module

from ..block import BlockTestBase, TestBlock, link


class TestGrapher(BlockTestBase):
  """Unit tests for the Grapher Block-specific GUI behavior."""

  def setUp(self) -> None:
    """Tracks Grapher Blocks and closes stale Matplotlib figures."""

    self._graphers: list[Grapher] = list()
    grapher_module.plt.close('all')

  def tearDown(self) -> None:
    """Closes Matplotlib figures before resetting the Block class state."""

    for grapher in self._graphers:
      grapher.finish()
    grapher_module.plt.close('all')

    super().tearDown()

  def _prepare_grapher(self,
                       *labels: tuple[str, str],
                       **kwargs) -> tuple[Grapher, TestBlock]:
    """Creates a linked Grapher and prepares its Matplotlib figure."""

    kwargs.setdefault('backend', 'Agg')

    source = TestBlock()
    grapher = Grapher(*labels, **kwargs)
    link(source, grapher)

    with (patch.object(grapher_module.plt, 'show'),
          patch.object(grapher_module.plt, 'pause')):
      grapher.prepare()

    self._graphers.append(grapher)
    return grapher, source

  def test_labels_are_validated(self) -> None:
    """Checks that labels must be pairs."""

    with self.assertRaises(ValueError):
      Grapher(('x',), backend='Agg')

    with self.assertRaises(ValueError):
      Grapher(('x', 'y', 'z'), backend='Agg')

    self.assertEqual(Grapher(('x', 'y'), backend='Agg')._labels,
                     (('x', 'y'),))

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Grapher without input Links fails early."""

    grapher = Grapher(('x', 'y'), backend='Agg')

    with self.assertRaises(IOError):
      grapher.prepare()

  def test_prepare_builds_figure_and_lines(self) -> None:
    """Checks the Matplotlib figure and curve objects created by prepare."""

    grapher, _ = self._prepare_grapher(('x', 'y'),
                                       ('x', 'z'),
                                       window_size=(3, 2))

    self.assertIsNotNone(grapher._figure)
    self.assertIs(grapher._canvas, grapher._figure.canvas)
    self.assertIsNotNone(grapher._ax)
    self.assertEqual(len(grapher._lines), 2)
    self.assertEqual(grapher._factor, [1, 1])
    self.assertEqual(grapher._counter, [0, 0])
    self.assertEqual(grapher._ax.get_title(loc='right'),
                     '(Press c to clear the graph)')
    self.assertEqual(grapher._ax.get_xlabel(), 'x')
    self.assertEqual({text.get_text() for text in grapher._ax.get_legend().texts},
                     {'y', 'z'})

  def test_prepare_can_create_marker_only_curves(self) -> None:
    """Checks the non-interpolated display mode."""

    grapher, _ = self._prepare_grapher(('x', 'y'), interp=False)

    self.assertEqual(grapher._lines[0].get_marker(), 'o')
    self.assertEqual(grapher._lines[0].get_markersize(), 3)

  def test_loop_uses_nonblocking_receive_at_high_frequency(self) -> None:
    """Checks receive helper arguments for fast loop rates."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=10)
    calls = list()

    def recv_all_data_raw(delay=None, poll_delay=None):
      calls.append((delay, poll_delay))
      return []

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(calls, [(None, None)])

  def test_loop_uses_nonblocking_receive_without_frequency_limit(self) -> None:
    """Checks that freq=None loops as fast as possible."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=None)
    calls = list()

    def recv_all_data_raw(delay=None, poll_delay=None):
      calls.append((delay, poll_delay))
      return []

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(calls, [(None, None)])

  def test_loop_uses_timed_receive_at_low_frequency(self) -> None:
    """Checks receive helper arguments for slow loop rates."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=2)
    calls = list()

    def recv_all_data_raw(delay=None, poll_delay=None):
      calls.append((delay, poll_delay))
      return []

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(calls, [(0.25, 0.1)])

  def test_loop_updates_matching_curves_only(self) -> None:
    """Checks that only payloads with both requested labels update a curve."""

    grapher, _ = self._prepare_grapher(('x', 'y'), ('x', 'z'), freq=10)

    def recv_all_data_raw(delay=None, poll_delay=None):
      return [
        {'x': [1, 2], 'y': [3, 4]},
        {'x': [5], 'unused': [6]},
      ]

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 2])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [3, 4])
    self.assertEqual(list(grapher._lines[1].get_xdata()), [])
    self.assertEqual(list(grapher._lines[1].get_ydata()), [])

  def test_loop_keeps_only_requested_length(self) -> None:
    """Checks that length limits the number of displayed points."""

    grapher, _ = self._prepare_grapher(('x', 'y'), length=3, freq=10)

    def recv_all_data_raw(delay=None, poll_delay=None):
      return [{'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [2, 3, 4])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [6, 7, 8])

  def test_loop_resamples_when_max_points_is_exceeded(self) -> None:
    """Checks the max_pt resampling behavior and factor update."""

    grapher, _ = self._prepare_grapher(('x', 'y'), max_pt=3, freq=10)

    def recv_all_data_raw(delay=None, poll_delay=None):
      return [{'x': [1, 2, 3, 4, 5], 'y': [6, 7, 8, 9, 10]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 3, 5])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [6, 8, 10])
    self.assertEqual(grapher._factor, [2])
    self.assertEqual(grapher._counter, [0])

  def test_loop_ignores_tcl_errors_while_drawing(self) -> None:
    """Checks that draw and event flushing errors are tolerated."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=10)

    def recv_all_data_raw(delay=None, poll_delay=None):
      return [{'x': [1], 'y': [2]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    with patch.object(grapher._canvas, 'flush_events', side_effect=TclError):
      grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [2])

  def test_clear_key_resets_lines_and_resampling_state(self) -> None:
    """Checks the c-key callback behavior."""

    grapher, _ = self._prepare_grapher(('x', 'y'), ('x', 'z'))
    for line in grapher._lines:
      line.set_xdata([1, 2])
      line.set_ydata([3, 4])
    grapher._factor = [2, 4]
    grapher._counter = [1, 3]
    event = type('Event', (), {'key': 'c'})()

    grapher._on_press(event)

    for line in grapher._lines:
      self.assertEqual(list(line.get_xdata()), [])
      self.assertEqual(list(line.get_ydata()), [])
    self.assertEqual(grapher._factor, [1, 1])
    self.assertEqual(grapher._counter, [0, 0])

  def test_other_keys_do_not_clear_graph(self) -> None:
    """Checks that only the c key clears the graph."""

    grapher, _ = self._prepare_grapher(('x', 'y'))
    grapher._lines[0].set_xdata([1])
    grapher._lines[0].set_ydata([2])
    grapher._factor = [2]
    grapher._counter = [1]
    event = type('Event', (), {'key': 'x'})()

    grapher._on_press(event)

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [2])
    self.assertEqual(grapher._factor, [2])
    self.assertEqual(grapher._counter, [1])

  def test_finish_closes_only_its_own_figure(self) -> None:
    """Checks that finish does not close another current Matplotlib figure."""

    grapher_1, _ = self._prepare_grapher(('x', 'y'))
    grapher_2, _ = self._prepare_grapher(('x', 'y'))
    grapher_module.plt.figure(grapher_2._figure.number)

    grapher_1.finish()

    self.assertNotIn(grapher_1._figure.number, grapher_module.plt.get_fignums())
    self.assertIn(grapher_2._figure.number, grapher_module.plt.get_fignums())
