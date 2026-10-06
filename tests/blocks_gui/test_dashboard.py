# coding: utf-8

from tkinter import TclError
from unittest.mock import patch
import numpy as np
import logging
from crappy.blocks.dashboard import Dashboard, DashboardWindow
import crappy.blocks.dashboard as dashboard_module

from ..block import BlockTestBase, TestBlock, link


class DashboardTests:
  """Common value and formatting checks for both graphical backends."""

  def setUp(self) -> None:
    """Tracks Dashboard Blocks for cleanup, even if prepare fails."""

    super().setUp()
    patcher = patch.object(dashboard_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._dashboards: list[Dashboard] = list()

  def tearDown(self) -> None:
    """Closes windows before resetting the Block class state."""

    for dashboard in self._dashboards:
      dashboard.finish()

    super().tearDown()

  def _prepare_dashboard(self, labels, **kwargs) -> tuple[Dashboard,
                                                          TestBlock]:
    """Creates a linked Dashboard and prepares its GUI."""

    kwargs.setdefault('backend', self._backend)
    source = TestBlock()
    dashboard = Dashboard(labels, **kwargs)
    link(source, dashboard)

    self._dashboards.append(dashboard)
    dashboard.prepare()
    if self._backend == 'tkinter':
      dashboard._dashboard.withdraw()
    return dashboard, source

  def _value(self, dashboard: Dashboard, label: str) -> str:
    """Read a displayed value using the selected backend's native widget."""

    return (dashboard._dashboard.tk_var[label].get()
            if self._backend == 'tkinter'
            else dashboard._qt_values[label].text())

  def test_labels_are_normalized(self) -> None:
    """Checks the supported label argument forms."""

    self.assertEqual(Dashboard('abc')._dash_labels, ['abc'])
    self.assertEqual(Dashboard(('a', 'b'))._dash_labels, ['a', 'b'])

  def test_nb_digits_is_validated(self) -> None:
    """Checks that invalid decimal precision values are rejected early."""

    for nb_digits, error in ((-1, ValueError), (1.5, TypeError),
                              ('2', TypeError), (True, TypeError)):
      with self.subTest(nb_digits=nb_digits):
        with self.assertRaises(error):
          Dashboard('a', nb_digits=nb_digits)

    self.assertEqual(Dashboard('a', nb_digits=0)._nb_digits, 0)
    self.assertEqual(Dashboard('a', nb_digits=3)._nb_digits, 3)

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Dashboard without input Links fails early."""

    dashboard = Dashboard('a')

    with self.assertRaises(IOError):
      dashboard.prepare()

  def test_prepare_creates_dashboard_window(self) -> None:
    """Checks the two-column window and its initial empty values."""

    dashboard, _ = self._prepare_dashboard(('a', 'b'))
    if self._backend == 'pyqt':
      window = dashboard._qt_window
      self.assertEqual(window.windowTitle(), 'Dashboard')
      self.assertIsNone(dashboard._dashboard)
      self.assertEqual(list(dashboard._qt_labels), ['a', 'b'])
      self.assertEqual(list(dashboard._qt_values), ['a', 'b'])
      self.assertTrue(window.font().bold())
      self.assertEqual(window.font().pointSize(), 48)
      constraints = dashboard_module.QtWidgets.QLayout.SizeConstraint
      self.assertEqual(window.layout().sizeConstraint(),
                       constraints.SetFixedSize)
      self.assertEqual(window.styleSheet(), '')
      for row, label in enumerate(('a', 'b')):
        self.assertEqual(dashboard._qt_labels[label].text(), f'{label}:')
        self.assertEqual(self._value(dashboard, label), '')
        self.assertIs(window.layout().itemAtPosition(row, 0).widget(),
                       dashboard._qt_labels[label])
        self.assertIs(window.layout().itemAtPosition(row, 1).widget(),
                       dashboard._qt_values[label])
      return

    window = dashboard._dashboard
    self.assertIsInstance(window, DashboardWindow)
    self.assertEqual(window.title(), 'Dashboard')
    self.assertEqual(window._labels, ['a', 'b'])
    self.assertEqual(set(window.tk_var), {'a', 'b'})
    self.assertEqual(set(window._tk_labels), {'a', 'b'})
    self.assertEqual(set(window._tk_values), {'a', 'b'})

    for label in ('a', 'b'):
      with self.subTest(label=label):
        self.assertEqual(window.tk_var[label].get(), '')
        self.assertEqual(window._tk_labels[label].cget('text'),
                         f'{label}:')
        self.assertEqual(str(window._tk_values[label].cget('textvariable')),
                         str(window.tk_var[label]))

  def test_loop_displays_latest_requested_values(self) -> None:
    """Checks string and numeric formatting for requested labels."""

    dashboard, source = self._prepare_dashboard(('name', 'value', 'count'),
                                                nb_digits=2)

    source.send({'name': 'first',
                 'value': 1.234,
                 'count': np.int64(3),
                 'ignored': 10})
    source.send({'name': 'last',
                 'value': 5.678,
                 'count': np.int64(4),
                 'ignored': 20})

    dashboard.loop()

    self.assertEqual(self._value(dashboard, 'name'), 'last')
    self.assertEqual(self._value(dashboard, 'value'), '5.68')
    self.assertEqual(self._value(dashboard, 'count'), '4.00')
    displayed = (dashboard._dashboard.tk_var if self._backend == 'tkinter'
                 else dashboard._qt_values)
    self.assertNotIn('ignored', displayed)

  def test_loop_respects_decimal_precision(self) -> None:
    """Checks decimal precision, including integer display."""

    dashboard, source = self._prepare_dashboard(('value',), nb_digits=0)

    source.send({'value': 1.6})

    dashboard.loop()

    self.assertEqual(self._value(dashboard, 'value'), '2')

  def test_loop_does_not_fill_missing_values(self) -> None:
    """Checks that only values received during the current loop update."""

    dashboard, source = self._prepare_dashboard(('a', 'b'))

    source.send({'a': 1})
    dashboard.loop()

    self.assertEqual(self._value(dashboard, 'a'), '1.00')
    self.assertEqual(self._value(dashboard, 'b'), '')

    source.send({'b': 2})
    dashboard.loop()

    self.assertEqual(self._value(dashboard, 'a'), '1.00')
    self.assertEqual(self._value(dashboard, 'b'), '2.00')

  def test_loop_warns_on_unsupported_values(self) -> None:
    """Checks that unsupported requested values are ignored and logged."""

    dashboard, source = self._prepare_dashboard(('a',))
    logs = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    dashboard.log = log
    value = object()
    source.send({'a': value})

    dashboard.loop()

    self.assertEqual(self._value(dashboard, 'a'), '')
    warning_logs = [msg for level, msg in logs if level == logging.WARNING]
    self.assertEqual(len(warning_logs), 1)
    self.assertIn("Don't know how to handle the received value",
                  warning_logs[0])

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes the selected backend's window."""

    dashboard, _ = self._prepare_dashboard(('a',))

    dashboard.finish()

    if self._backend == 'tkinter':
      with self.assertRaises(TclError):
        dashboard._dashboard.wm_state()
    else:
      self.assertFalse(dashboard._qt_window.isVisible())

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    dashboard, _ = self._prepare_dashboard(('a',))

    dashboard.finish()
    dashboard.finish()

  def test_loop_services_gui_without_new_data(self) -> None:
    """An idle Dashboard remains responsive without changing its values."""

    dashboard, _ = self._prepare_dashboard(('a',))
    target = (dashboard._dashboard if self._backend == 'tkinter'
              else dashboard._qt_app)
    method = 'update' if self._backend == 'tkinter' else 'processEvents'
    with patch.object(target, method) as update:
      dashboard.loop()
    update.assert_called_once_with()
    self.assertEqual(self._value(dashboard, 'a'), '')


class TestDashboard(DashboardTests, BlockTestBase):
  """Legacy Tkinter Dashboard integration."""

  _backend = 'tkinter'

  def test_loop_ignores_update_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    dashboard, source = self._prepare_dashboard(('a',))
    source.send({'a': 1})
    with patch.object(dashboard._dashboard, 'update', side_effect=TclError):
      dashboard.loop()
    self.assertEqual(self._value(dashboard, 'a'), '1.00')


class TestDashboardPyQt(DashboardTests, BlockTestBase):
  """Native Qt Dashboard layout, rendering, and application reuse."""

  _backend = 'pyqt'

  def test_labels_and_received_strings_use_plain_text(self) -> None:
    """Names and values resembling HTML must be displayed literally."""

    label = '<b>value</b>'
    dashboard, source = self._prepare_dashboard((label,))
    source.send({label: '<i>text</i>'})
    dashboard.loop()
    self.assertEqual(self._value(dashboard, label), '<i>text</i>')
    for widget in (dashboard._qt_labels[label], dashboard._qt_values[label]):
      self.assertEqual(widget.textFormat(),
                       dashboard_module.QtCore.Qt.TextFormat.PlainText)

  def test_closing_window_does_not_stop_test_or_other_windows(self) -> None:
    """Closing one Dashboard leaves the application and its peers alive."""

    dashboard, _ = self._prepare_dashboard(('a',))
    other, _ = self._prepare_dashboard(('b',))
    self.assertIs(dashboard._qt_app, other._qt_app)
    with patch.object(dashboard, 'stop') as stop:
      dashboard._qt_window.close()
      dashboard.loop()
      stop.assert_not_called()
    dashboard.finish()
    self.assertTrue(other._qt_window.isVisible())
