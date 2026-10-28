from tests.support.assertions import with_failure_manager
from unittest.mock import patch
import numpy as np
import logging
from crappy.blocks.dashboard import Dashboard
import crappy.blocks.dashboard as dashboard_module
from tests.support.blocks import TestBlock, link


class DashboardTests:
  """Common value and formatting checks for both graphical backends."""

  def setUp(self) -> None:
    """Tracks Dashboard Blocks for cleanup, even if prepare fails."""

    super().setUp()
    patcher = patch.object(dashboard_module, 'warn')
    patcher.start()
    self.addCleanup(patcher.stop)
    self._dashboards: list[Dashboard] = list()

  def _prepare_dashboard(self, labels, **kwargs) -> tuple[Dashboard,
                                                          TestBlock]:
    """Creates a linked Dashboard and prepares its GUI."""

    kwargs.setdefault('backend', self._backend)
    source = TestBlock()
    dashboard = with_failure_manager(Dashboard(labels, **kwargs))
    self.addCleanup(dashboard.finish)
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
    """String and sequence labels select the values displayed to the user."""

    for labels in ('abc', ('a', 'b')):
      with self.subTest(labels=labels):
        expected = [labels] if isinstance(labels, str) else labels
        dashboard, source = self._prepare_dashboard(labels, nb_digits=0)
        source.send({name: index + 1 for index, name in enumerate(expected)})
        dashboard.loop()
        for index, name in enumerate(expected):
          self.assertEqual(self._value(dashboard, name), str(index + 1))

  def test_nb_digits_is_validated(self) -> None:
    """Invalid precision is rejected; valid values control the displayed digits."""

    for nb_digits in (-1, 1.5, '2', True):
      with self.subTest(nb_digits=nb_digits):
        with self.assertRaises((TypeError, ValueError)):
          Dashboard('a', nb_digits=nb_digits)
    for digits, expected in ((0, '1'), (3, '1.235')):
      with self.subTest(digits=digits):
        dashboard, source = self._prepare_dashboard('a', nb_digits=digits)
        source.send({'a': 1.23456})
        dashboard.loop()
        self.assertEqual(self._value(dashboard, 'a'), expected)

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Dashboard without input Links fails early."""

    dashboard = with_failure_manager(Dashboard('a'))
    self.addCleanup(dashboard.finish)

    with self.assertRaises(IOError):
      dashboard.prepare()

  def test_prepare_creates_dashboard_window(self) -> None:
    """A new dashboard is usable and has no received values yet."""

    dashboard, _ = self._prepare_dashboard(('a', 'b'))
    for label in ('a', 'b'):
      self.assertEqual(self._value(dashboard, label), '')
    if self._backend == 'tkinter':
      self.assertTrue(dashboard._dashboard.winfo_exists())
    else:
      self.assertTrue(dashboard._qt_window.isVisible())

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
    self.assertTrue(warning_logs)

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes the selected backend's window."""

    dashboard, _ = self._prepare_dashboard(('a',))
    window = (dashboard._dashboard if self._backend == 'tkinter'
              else dashboard._qt_window)

    dashboard.finish()

    if self._backend == 'tkinter':
      from tkinter import TclError
      with self.assertRaises(TclError):
        window.wm_state()
    else:
      self.assertFalse(window.isVisible())

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    dashboard, _ = self._prepare_dashboard(('a',))

    dashboard.finish()
    dashboard.finish()

  def test_loop_services_gui_without_new_data(self) -> None:
    """An idle Dashboard processes queued UI work without changing its values."""

    dashboard, _ = self._prepare_dashboard(('a',))
    handled = []
    if self._backend == 'tkinter':
      dashboard._dashboard.after_idle(lambda: handled.append(True))
    else:
      from PyQt6.QtCore import QTimer
      QTimer.singleShot(0, lambda: handled.append(True))
    dashboard.loop()
    self.assertEqual(handled, [True])
    self.assertEqual(self._value(dashboard, 'a'), '')
