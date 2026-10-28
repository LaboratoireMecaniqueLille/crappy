from tests.support.gui import require_gui_group

require_gui_group('pyqt')

from tests.support.assertions import with_failure_manager
from unittest.mock import patch
from matplotlib.backend_bases import KeyEvent
from tests.support.clock import Clock
from crappy.blocks.grapher import Grapher
import crappy.blocks.grapher as grapher_module
from tests.support.blocks import TestBlock, link

from tests.gui._support.grapher import GrapherTestBase


class TestGrapher(GrapherTestBase):
  def test_labels_are_validated(self) -> None:
    """Checks that labels must be non-empty string pairs."""

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher())

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x',)))

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y', 'z')))

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', '')))

    grapher, _ = self._prepare_grapher(('x', 'y'))
    grapher.recv_all_data_raw = lambda: [{'x': [1], 'y': [2]}]
    grapher.loop()
    self.assertEqual(list(grapher._lines[0].get_xdata()), [1])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [2])

  def test_constructor_validates_plotter_and_backend(self) -> None:
    """Checks plotter selection and plotter-specific backends."""

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y'), plotter='invalid'))
    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y'), backend='TkAgg'))
    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y'), plotter='mpl', backend='PyQt6'))

  def test_constructor_validates_limits_and_refresh_frequency(self) -> None:
    """Checks validation of the new buffering and refresh arguments."""

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y'), length=3))

    with self.assertRaises(ValueError):
      with_failure_manager(Grapher(('x', 'y'), upd_freq=11, freq=10))

    for options in ({'length': True, 'max_pt': None}, {'max_pt': True}):
      with self.subTest(options=options):
        grapher, _ = self._prepare_grapher(('x', 'y'), upd_freq=True, **options)
        grapher.recv_all_data_raw = lambda: [{'x': [1, 2, 3], 'y': [4, 5, 6]}]
        grapher.loop()
        xs = list(grapher._lines[0].get_xdata())
        ys = list(grapher._lines[0].get_ydata())
        self.assertEqual(len(xs), 1)
        self.assertEqual(ys, [value + 3 for value in xs])

    for value in (float('nan'), float('inf')):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          with_failure_manager(Grapher(('x', 'y'), upd_freq=value, freq=None))

  def test_constructor_accepts_fractional_size_and_screen_coordinates(
      self) -> None:
    """Checks useful Matplotlib sizes and multi-screen positions."""

    grapher, _ = self._prepare_grapher(
      ('x', 'y'), window_size=(3.5, 2.25), window_pos=(-100, 0))

    self.assertEqual(tuple(grapher._figure.get_size_inches()), (3.5, 2.25))

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Grapher without input Links fails early."""

    grapher = with_failure_manager(Grapher(('x', 'y')))

    with self.assertRaises(IOError):
      grapher.prepare()

  def test_prepare_rejects_output_links(self) -> None:
    """Checks that a Grapher cannot be used as a data source."""

    source = TestBlock()
    grapher = with_failure_manager(Grapher(('x', 'y')))
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
    self.assertEqual(tuple(grapher._figure.get_size_inches()), (3.5, 2.25))
    self.assertEqual(set(grapher._ax.get_xlabel().split(', ')), {'x', 'time'})
    self.assertEqual(set(grapher._ax.get_ylabel().split(', ')), {'y', 'z'})
    self.assertEqual({text.get_text() for text in grapher._ax.get_legend().texts},
                     {'y', 'z'})

  def test_prepare_can_create_marker_only_curves(self) -> None:
    """Checks the non-interpolated display mode."""

    grapher, _ = self._prepare_grapher(('x', 'y'), interp=False)

    self.assertNotIn(grapher._lines[0].get_marker(), ('None', 'none', '', None))
    self.assertIn(grapher._lines[0].get_linestyle(), ('None', 'none', '', None))

  def test_prepare_builds_pyqtgraph_window_and_curves(self) -> None:
    """Checks the PyQtGraph application, plot, and curve objects."""

    grapher, _ = self._prepare_pyqt_grapher(
        ('x', 'y'), ('time', 'z'), window_size=(3.5, 2.25))

    self.assertIsNotNone(grapher._qt_app)
    self.assertIsNotNone(grapher._qt_plot)
    self.assertEqual(len(grapher._qt_curves), 2)
    self.assertEqual({curve.name() for curve in grapher._qt_curves},
                     {'y', 'z'})

  def test_pyqtgraph_ignores_opencv_qt_plugin_path(self) -> None:
    """Checks OpenCV's Qt plugins are hidden while creating the Qt app."""

    variable = "QT_QPA_PLATFORM_PLUGIN_PATH"
    opencv_path = grapher_module.os.path.join(
        'somewhere', 'cv2', 'qt', 'plugins')
    unrelated_path = grapher_module.os.path.join(
        'somewhere', 'other', 'plugins')

    cases = (
      (opencv_path, None),
      (grapher_module.os.pathsep.join((unrelated_path, opencv_path)),
       unrelated_path),
    )

    for original_path, expected_path in cases:
      with self.subTest(original_path=original_path):
        observed_paths = list()
        qt_app = object()

        def make_qt_app(*_) -> object:
          observed_paths.append(grapher_module.os.environ.get(variable))
          return qt_app

        grapher = with_failure_manager(Grapher(('x', 'y')))
        with (patch.dict(grapher_module.os.environ,
                         {variable: original_path}),
              patch.object(grapher_module, 'pg') as pyqtgraph):
          pyqtgraph.mkQApp.side_effect = make_qt_app
          grapher._handle_qt_backend_conflict()

          self.assertEqual(observed_paths, [expected_path])
          self.assertEqual(grapher_module.os.environ[variable], original_path)
          self.assertIs(grapher._qt_app, qt_app)

  def test_pyqtgraph_loop_updates_curve_data(self) -> None:
    """Checks that received values are forwarded to PyQtGraph curves."""

    grapher, _ = self._prepare_pyqt_grapher(('x', 'y'), upd_freq=None)
    grapher.recv_all_data_raw = lambda: [{'x': [1, 2], 'y': [3, 4]}]

    grapher.loop()

    x_data, y_data = grapher._qt_curves[0].getData()
    self.assertEqual(x_data.tolist(), [1, 2])
    self.assertEqual(y_data.tolist(), [3, 4])

  def test_pyqtgraph_uses_requested_binding(self) -> None:
    """Checks that the selected Qt binding reaches PyQtGraph."""

    with patch.dict(grapher_module.os.environ):
      self._prepare_pyqt_grapher(('x', 'y'), backend='PyQt6')

      self.assertEqual(grapher_module.os.environ['PYQTGRAPH_QT_LIB'], 'PyQt6')

  def test_pyqtgraph_updates_curves_and_handles_queued_events(self) -> None:
    """Curve updates keep the GUI event queue responsive."""

    grapher, _ = self._prepare_pyqt_grapher(
        ('x', 'y'), ('x', 'z'), upd_freq=None)
    from pyqtgraph.Qt import QtCore
    grapher.recv_all_data_raw = lambda: [{'x': [1, 2], 'y': [3, 4], 'z': [5, 6]}]
    callbacks = []
    QtCore.QTimer.singleShot(0, lambda: callbacks.append('handled'))

    grapher.loop()

    self.assertEqual(callbacks, ['handled'])
    for curve, expected in zip(grapher._qt_curves, ([3, 4], [5, 6])):
      xs, ys = curve.getData()
      self.assertEqual(xs.tolist(), [1, 2])
      self.assertEqual(ys.tolist(), expected)

  def test_loop_uses_nonblocking_receive(self) -> None:
    """Checks that receiving data never delays the independent loop timer."""

    grapher, _ = self._prepare_grapher(('x', 'y'), freq=2,
                                       upd_freq=2)
    calls = list()

    def recv_all_data_raw():
      calls.append(None)
      return []

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(calls, [None])

  def test_loop_updates_matching_curves_only(self) -> None:
    """Checks that only payloads with both requested labels update a curve."""

    grapher, _ = self._prepare_grapher(('x', 'y'), ('x', 'z'), freq=10)

    def recv_all_data_raw():
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
    """Refresh limiting retains all samples received since the last redraw."""

    clock = Clock(10)
    with patch.object(grapher_module, 'monotonic', clock):
      grapher, _ = self._prepare_grapher(('x', 'y'), upd_freq=2)
      grapher.recv_all_data_raw = lambda: []
      grapher.loop()
      chunks = iter((
        [{'x': [1], 'y': [3]}],
        [{'x': [2], 'y': [4]}],
      ))
      grapher.recv_all_data_raw = lambda: next(chunks)
      clock.advance(0.1)
      grapher.loop()
      self.assertEqual(list(grapher._lines[0].get_xdata()), [])
      clock.advance(0.5)
      grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 2])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [3, 4])

  def test_loop_keeps_only_requested_length(self) -> None:
    """Checks that length limits the number of displayed points."""

    grapher, _ = self._prepare_grapher(('x', 'y'), length=3, max_pt=None,
                                       freq=10)

    def recv_all_data_raw():
      return [{'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [2, 3, 4])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [6, 7, 8])

  def test_loop_resamples_when_max_points_is_exceeded(self) -> None:
    """Checks the max_pt resampling behavior and factor update."""

    grapher, _ = self._prepare_grapher(('x', 'y'), max_pt=3, freq=10)

    def recv_all_data_raw():
      return [{'x': [1, 2, 3, 4, 5], 'y': [6, 7, 8, 9, 10]}]

    grapher.recv_all_data_raw = recv_all_data_raw

    grapher.loop()

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 3, 5])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [6, 8, 10])

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

  def test_clear_key_resets_lines_and_resampling_state(self) -> None:
    """Clear removes old samples and restarts sampling for new data."""

    grapher, _ = self._prepare_grapher(('x', 'y'), ('x', 'z'),
                                      max_pt=3, upd_freq=None)
    chunks = iter((
      [{'x': [1, 2, 3, 4, 5], 'y': [11, 12, 13, 14, 15], 'z': [21, 22, 23, 24, 25]}],
      [{'x': [6, 7], 'y': [16, 17], 'z': [26, 27]}],
    ))
    grapher.recv_all_data_raw = lambda: next(chunks)
    grapher.loop()
    event = KeyEvent('key_press_event', grapher._canvas, key='c')

    grapher._canvas.callbacks.process('key_press_event', event)

    for line in grapher._lines:
      self.assertEqual(list(line.get_xdata()), [])
      self.assertEqual(list(line.get_ydata()), [])
    grapher.loop()
    for line, expected in zip(grapher._lines, ([16, 17], [26, 27])):
      self.assertEqual(list(line.get_xdata()), [6, 7])
      self.assertEqual(list(line.get_ydata()), expected)

  def test_other_keys_do_not_clear_graph(self) -> None:
    """An unrelated key preserves displayed data and sampling progression."""

    grapher, _ = self._prepare_grapher(('x', 'y'), max_pt=3, upd_freq=None)
    chunks = iter((
      [{'x': [1, 2, 3, 4, 5], 'y': [11, 12, 13, 14, 15]}],
      [{'x': [6, 7, 8, 9], 'y': [16, 17, 18, 19]}],
    ))
    grapher.recv_all_data_raw = lambda: next(chunks)
    grapher.loop()
    event = KeyEvent('key_press_event', grapher._canvas, key='x')

    grapher._canvas.callbacks.process('key_press_event', event)

    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 3, 5])
    grapher.loop()
    self.assertEqual(list(grapher._lines[0].get_xdata()), [1, 5, 9])
    self.assertEqual(list(grapher._lines[0].get_ydata()), [11, 15, 19])

  def test_finish_closes_only_its_own_figure(self) -> None:
    """Checks that finish does not close another current Matplotlib figure."""

    grapher_1, _ = self._prepare_grapher(('x', 'y'))
    grapher_2, _ = self._prepare_grapher(('x', 'y'))
    grapher_module.plt.figure(grapher_2._figure.number)
    figure_number = grapher_1._figure.number

    grapher_1.finish()

    self.assertNotIn(figure_number, grapher_module.plt.get_fignums())
    self.assertIn(grapher_2._figure.number, grapher_module.plt.get_fignums())

  def test_clear_resets_pyqtgraph_curves_and_resampling_state(self) -> None:
    """The Qt clear shortcut removes prior data and restarts sampling."""

    grapher, _ = self._prepare_pyqt_grapher(('x', 'y'), max_pt=3, upd_freq=None)
    chunks = iter((
      [{'x': [1, 2, 3, 4, 5], 'y': [11, 12, 13, 14, 15]}],
      [{'x': [6, 7], 'y': [16, 17]}],
    ))
    grapher.recv_all_data_raw = lambda: next(chunks)
    grapher.loop()

    grapher._on_press_pyqt()

    xs, ys = grapher._qt_curves[0].getData()
    self.assertTrue(xs is None or not len(xs))
    self.assertTrue(ys is None or not len(ys))
    grapher.loop()
    xs, ys = grapher._qt_curves[0].getData()
    self.assertEqual(xs.tolist(), [6, 7])
    self.assertEqual(ys.tolist(), [16, 17])

  def test_finish_before_prepare_does_not_close_another_figure(self) -> None:
    """Checks that an unprepared Grapher does not close a current figure."""

    figure = grapher_module.plt.figure()
    grapher = with_failure_manager(Grapher(('x', 'y'), plotter='mpl'))

    grapher.finish()

    self.assertIn(figure.number, grapher_module.plt.get_fignums())
