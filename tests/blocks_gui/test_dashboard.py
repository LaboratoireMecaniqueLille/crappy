# coding: utf-8

from tkinter import TclError
from unittest.mock import patch
import numpy as np
import logging
from crappy.blocks.dashboard import Dashboard, DashboardWindow

from ..block import BlockTestBase, TestBlock, link


class TestDashboard(BlockTestBase):
  """Unit tests for the Dashboard Block-specific GUI behavior."""

  def setUp(self) -> None:
    """Tracks Dashboard Blocks to destroy their Tk windows."""

    self._dashboards: list[Dashboard] = list()

  def tearDown(self) -> None:
    """Destroys Tk windows before resetting the Block class state."""

    for dashboard in self._dashboards:
      dashboard.finish()

    super().tearDown()

  def _prepare_dashboard(self, labels, **kwargs) -> tuple[Dashboard,
                                                          TestBlock]:
    """Creates a linked Dashboard and prepares its GUI."""

    source = TestBlock()
    dashboard = Dashboard(labels, **kwargs)
    link(source, dashboard)

    dashboard.prepare()
    dashboard._dashboard.withdraw()

    self._dashboards.append(dashboard)
    return dashboard, source

  def test_labels_are_normalized(self) -> None:
    """Checks the supported label argument forms."""

    self.assertEqual(Dashboard('abc')._labels, ['abc'])
    self.assertEqual(Dashboard(('a', 'b'))._labels, ['a', 'b'])

  def test_nb_digits_is_validated(self) -> None:
    """Checks that invalid decimal precision values are rejected early."""

    for nb_digits in (-1, 1.5, '2'):
      with self.subTest(nb_digits=nb_digits):
        with self.assertRaises(ValueError):
          Dashboard('a', nb_digits=nb_digits)

    self.assertEqual(Dashboard('a', nb_digits=0)._nb_digits, 0)
    self.assertEqual(Dashboard('a', nb_digits=3)._nb_digits, 3)

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Dashboard without input Links fails early."""

    dashboard = Dashboard('a')

    with self.assertRaises(IOError):
      dashboard.prepare()

  def test_prepare_creates_dashboard_window(self) -> None:
    """Checks the Tk window and widgets created by prepare."""

    dashboard, _ = self._prepare_dashboard(('a', 'b'))
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

    self.assertEqual(dashboard._dashboard.tk_var['name'].get(), 'last')
    self.assertEqual(dashboard._dashboard.tk_var['value'].get(), '5.68')
    self.assertEqual(dashboard._dashboard.tk_var['count'].get(), '4.00')
    self.assertNotIn('ignored', dashboard._dashboard.tk_var)

  def test_loop_respects_decimal_precision(self) -> None:
    """Checks decimal precision, including integer display."""

    dashboard, source = self._prepare_dashboard(('value',), nb_digits=0)

    source.send({'value': 1.6})

    dashboard.loop()

    self.assertEqual(dashboard._dashboard.tk_var['value'].get(), '2')

  def test_loop_does_not_fill_missing_values(self) -> None:
    """Checks that only values received during the current loop update."""

    dashboard, source = self._prepare_dashboard(('a', 'b'))

    source.send({'a': 1})
    dashboard.loop()

    self.assertEqual(dashboard._dashboard.tk_var['a'].get(), '1.00')
    self.assertEqual(dashboard._dashboard.tk_var['b'].get(), '')

    source.send({'b': 2})
    dashboard.loop()

    self.assertEqual(dashboard._dashboard.tk_var['a'].get(), '1.00')
    self.assertEqual(dashboard._dashboard.tk_var['b'].get(), '2.00')

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

    self.assertEqual(dashboard._dashboard.tk_var['a'].get(), '')
    warning_logs = [msg for level, msg in logs if level == logging.WARNING]
    self.assertEqual(len(warning_logs), 1)
    self.assertIn("Don't know how to handle the received value",
                  warning_logs[0])

  def test_loop_ignores_update_tcl_errors(self) -> None:
    """Checks that loop tolerates Tk update errors."""

    dashboard, source = self._prepare_dashboard(('a',))
    source.send({'a': 1})

    with patch.object(dashboard._dashboard, 'update', side_effect=TclError):
      dashboard.loop()

    self.assertEqual(dashboard._dashboard.tk_var['a'].get(), '1.00')

  def test_finish_destroys_window(self) -> None:
    """Checks that finish closes the Tk window."""

    dashboard, _ = self._prepare_dashboard(('a',))

    dashboard.finish()

    with self.assertRaises(TclError):
      dashboard._dashboard.wm_state()

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called after the window is already gone."""

    dashboard, _ = self._prepare_dashboard(('a',))

    dashboard.finish()
    dashboard.finish()
