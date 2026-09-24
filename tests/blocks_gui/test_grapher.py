# coding: utf-8

from tkinter import TclError
from unittest.mock import patch

from matplotlib import pyplot as mpl_plt

from crappy.blocks.grapher import Grapher
import crappy.blocks.grapher as grapher_module

from ..block import BlockTestBase, TestBlock, link


class TestGrapher(BlockTestBase):
  """Unit tests for the Grapher Block-specific GUI behavior."""

  def setUp(self) -> None:
    """Tracks Grapher Blocks and closes stale Matplotlib figures."""

    super().setUp()
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

    with (patch.object(mpl_plt, 'show'),
          patch.object(mpl_plt, 'pause')):
      grapher.prepare()

    self._graphers.append(grapher)
    return grapher, source

  def test_labels_are_validated(self) -> None:
    """Checks that labels must be non-empty string pairs."""

    with self.assertRaises(ValueError):
      Grapher(backend='Agg')

    with self.assertRaises(ValueError):
      Grapher(('x',), backend='Agg')

    with self.assertRaises(ValueError):
      Grapher(('x', 'y', 'z'), backend='Agg')

    with self.assertRaises(ValueError):
      Grapher(('x', ''), backend='Agg')

    self.assertEqual(Grapher(('x', 'y'), backend='Agg')._graph_labels,
                     (('x', 'y'),))

  def test_constructor_validates_limits_and_refresh_frequency(self) -> None:
    """Checks validation of the new buffering and refresh arguments."""

    with self.assertRaises(ValueError):
      Grapher(('x', 'y'), length=3, backend='Agg')

    with self.assertRaises(ValueError):
      Grapher(('x', 'y'), upd_freq=11, freq=10, backend='Agg')

    self.assertEqual(Grapher(('x', 'y'), length=True, max_pt=None,
                             backend='Agg')._length, 1)
    self.assertEqual(Grapher(('x', 'y'), max_pt=True,
                             backend='Agg')._max_pt, 1)
    self.assertEqual(Grapher(('x', 'y'), upd_freq=True,
                             backend='Agg')._upd_freq, 1.0)

    for value in (float('nan'), float('inf')):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          Grapher(('x', 'y'), upd_freq=value, freq=None, backend='Agg')

  def test_constructor_accepts_fractional_size_and_screen_coordinates(
      self) -> None:
    """Checks useful Matplotlib sizes and multi-screen positions."""

    grapher, _ = self._prepare_grapher(
      ('x', 'y'), window_size=(3.5, 2.25), window_pos=(-100, 0))

    self.assertEqual(grapher._window_size, (3.5, 2.25))
    self.assertEqual(grapher._window_pos, (-100, 0))

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Grapher without input Links fails early."""

    grapher = Grapher(('x', 'y'), backend='Agg')

    with self.assertRaises(IOError):
      grapher.prepare()

  def test_prepare_rejects_output_links(self) -> None:
    """Checks that a Grapher cannot be used as a data source."""

    source = TestBlock()
    grapher = Grapher(('x', 'y'), backend='Agg')
    sink = TestBlock()
    link(source, grapher)
    link(grapher, sink)

    with self.assertRaises(IOError):
      grapher.prepare()

  def test_prepare_builds_figure_and_lines(self) -> None:
    """Checks the Matplotlib figure and curve objects created by prepare."""

    grapher, _ = self._prepare_grapher(('x', 'y'),
                                       ('time', 'z'),
                                       window_size=(3.5, 2.25))

    self.assertIsNotNone(grapher._figure)
    self.assertIs(grapher._canvas, grapher._figure.canvas)
    self.assertIsNotNone(grapher._ax)
    self.assertEqual(len(grapher._lines), 2)
    self.assertEqual(grapher._factor, [1, 1])
    self.assertEqual(grapher._counter, [0, 0])
    self.assertEqual(grapher._ax.get_title(loc='right'),
                     '(Press c to clear the graph)')
    self.assertEqual(tuple(grapher._figure.get_size_inches()), (3.5, 2.25))
    self.assertEqual(set(grapher._ax.get_xlabel().split(', ')), {'x', 'time'})
    self.assertEqual(set(grapher._ax.get_ylabel().split(', ')), {'y', 'z'})
    self.assertEqual({text.get_text() for text in grapher._ax.get_legend().texts},
                     {'y', 'z'})

  def test_prepare_can_create_marker_only_curves(self) -> None:
    """Checks the non-interpolated display mode."""

    grapher, _ = self._prepare_grapher(('x', 'y'), interp=False)

    self.assertEqual(grapher._lines[0].get_marker(), 'o')
    self.assertEqual(grapher._lines[0].get_markersize(), 3)

  def test_loop_uses_nonblocking_receive(self) -> None:
    """Checks that receiving data never delays the independent loop timer."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=2,
                                       upd_freq=2)
    calls = list()

    def recv_all_data_raw(delay=None, poll_delay=None):
      calls.append((delay, poll_delay))
      return []

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(calls, [(None, None)])

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

  def test_loop_uses_first_link_matching_a_curve(self) -> None:
    """Checks that one curve does not merge values from multiple Links."""

    grapher, _ = self._prepare_grapher(('x', 'y'), upd_freq=None)
    grapher.recv_all_data_raw = lambda: [
      {'x': [1, 2], 'y': [3, 4]},
      {'x': [5], 'y': [6]},
    ]

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 2])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [3, 4])

  def test_loop_rejects_mismatched_value_counts(self) -> None:
    """Checks that x and y values cannot become misaligned."""

    grapher, _ = self._prepare_grapher(('x', 'y'), upd_freq=None)
    grapher.recv_all_data_raw = lambda: [{'x': [1, 2], 'y': [3]}]

    with self.assertRaises(RuntimeError):
      grapher.loop()

  def test_loop_buffers_data_until_the_next_refresh(self) -> None:
    """Checks that refresh limiting does not discard received values."""

    grapher, _ = self._prepare_grapher(('x', 'y'), upd_freq=2)
    grapher._last_upd = 10
    chunks = iter((
      [{'x': [1], 'y': [3]}],
      [{'x': [2], 'y': [4]}],
    ))
    grapher.recv_all_data_raw = lambda: next(chunks)

    with patch('crappy.blocks.grapher.monotonic',
               side_effect=(10.1, 10.5, 10.5)):
      grapher.loop()
      self.assertEqual(list(grapher._lines[0].get_xdata()), [])
      grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 2])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [3, 4])

  def test_loop_keeps_only_requested_length(self) -> None:
    """Checks that length limits the number of displayed points."""

    grapher, _ = self._prepare_grapher(('x', 'y'), length=3, max_pt=None,
                                       freq=10)

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
    self.assertEqual(grapher._counter, [5])

  def test_loop_preserves_resampling_phase_between_chunks(self) -> None:
    """Checks that downsampling stays aligned across loop iterations."""

    grapher, _ = self._prepare_grapher(('x', 'y'), max_pt=3,
                                       upd_freq=None)
    chunks = iter((
      [{'x': [1, 2, 3, 4, 5], 'y': [11, 12, 13, 14, 15]}],
      [{'x': [6, 7, 8, 9], 'y': [16, 17, 18, 19]}],
    ))
    grapher.recv_all_data_raw = lambda: next(chunks)

    grapher.loop()
    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 5, 9])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [11, 15, 19])
    self.assertEqual(grapher._factor, [4])
    self.assertEqual(grapher._counter, [9])

  def test_loop_ignores_tcl_errors_while_drawing(self) -> None:
    """Checks that draw and event flushing errors are tolerated."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=10)

    def recv_all_data_raw(delay=None, poll_delay=None):
      return [{'x': [1], 'y': [2]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    canvas = grapher._canvas
    if canvas is None:
      self.fail("The Grapher canvas was not initialized")

    with patch.object(canvas, 'flush_events', side_effect=TclError):
      grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [2])

  def test_clear_key_resets_lines_and_resampling_state(self) -> None:
    """Checks the c-key callback behavior."""

    grapher, _ = self._prepare_grapher(('x', 'y'), ('x', 'z'))
    for line in grapher._lines:
      line.set_xdata([1, 2])
      line.set_ydata([3, 4])
    grapher._data[0][0].extend((1, 2))
    grapher._data[0][1].extend((3, 4))
    grapher._buf[0][0].append(5)
    grapher._buf[0][1].append(6)
    grapher._factor = [2, 4]
    grapher._counter = [1, 3]
    event = type('Event', (), {'key': 'c'})()

    grapher._on_press(event)

    for line in grapher._lines:
      self.assertEqual(list(line.get_xdata()), [])
      self.assertEqual(list(line.get_ydata()), [])
    self.assertEqual(grapher._factor, [1, 1])
    self.assertEqual(grapher._counter, [0, 0])
    self.assertTrue(all(not values for curve in grapher._data
                        for values in curve))
    self.assertTrue(all(not values for curve in grapher._buf
                        for values in curve))

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

  def test_finish_before_prepare_does_not_close_another_figure(self) -> None:
    """Checks that an unprepared Grapher does not close a current figure."""

    figure = grapher_module.plt.figure()
    grapher = Grapher(('x', 'y'), backend='Agg')

    grapher.finish()

    self.assertIn(figure.number, grapher_module.plt.get_fignums())
