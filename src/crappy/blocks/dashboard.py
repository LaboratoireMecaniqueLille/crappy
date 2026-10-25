# coding: utf-8

from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
import logging
import locale
import numbers
import os
from warnings import warn

from .meta_block import Block
from .._global import OptionalModule
from .. import _validation as check

tk = OptionalModule('tkinter', lazy_import=True)
QtCore = OptionalModule('PyQt6.QtCore', lazy_import=True)
QtWidgets = OptionalModule('PyQt6.QtWidgets', lazy_import=True)

__all__ = ('Dashboard', 'DashboardWindow')

if TYPE_CHECKING:
  import tkinter as tk
  from PyQt6 import QtWidgets, QtCore
  from ._dashboard_tkinter import DashboardWindow


class Dashboard(Block):
  """This Block generates an interface displaying data as text in a dedicated
  window.

  The graphical interface uses PyQt6 by default, or :mod:`tkinter` when
  requested. Both backends use the same two-column layout and large, bold
  text. Closing the window does not stop the test.

  In the window, the left column contains the names of the labels to display
  and the right column contains the latest received values for these labels.
  For each label, only the last value is therefore displayed. Strings are
  displayed as received, and real numbers are rounded to ``nb_digits`` decimal
  places. Missing or unsupported values leave the display unchanged.

  This Block provides a nicer display than the raw
  :class:`~crappy.blocks.LinkReader` Block. For displaying the evolution of a
  label over time, the :class:`~crappy.blocks.Grapher` Block should be used
  instead.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.1.0 the default backend is now PyQt, no longer Tkinter
  """

  def __init__(self,
               labels: str | Iterable[str],
               nb_digits: int = 2,
               backend: Literal['tkinter', 'pyqt'] = 'pyqt',
               freq: float | None = 30,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      labels: A non-empty string, or a non-empty iterable of non-empty
        strings. Only the data from these labels will be displayed, in the
        order provided.
      nb_digits: Non-negative integer number of decimals to show for real
        numbers. Strings are displayed without formatting.
      backend: The library used for the graphical interface, either
        ``'pyqt'`` (the default) or ``'tkinter'``. The ``'pyqt'`` backend
        requires PyQt6. Both backends display the label names on the left and
        their latest values on the right.

        .. versionadded:: 2.1.0
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.

        .. versionadded:: 1.5.7
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionadded:: 1.5.7
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0
    """

    warn("\nThe default backend for the Dashboard Block was changed from "
         "'tkinter' to 'pyqt' without prior notice.\nThis change was "
         "implemented nevertheless because it is part of a larger migration "
         "to PyQt\nSet backend='tkinter' to switch back to the "
         "previous behavior.\nInstall PyQt6 to use the new Dashboard "
         "interface.\n", UserWarning, stacklevel=2)

    self._dashboard: DashboardWindow | None = None
    self._qt_app: QtWidgets.QApplication | None = None
    self._qt_window: QtWidgets.QWidget | None = None
    self._qt_events_processed: bool = False

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    if isinstance(labels, str):
      labels = [labels]
    else:
      check.instance(labels, 'labels', Iterable)
      if isinstance(labels, (bytes, bytearray)):
        raise TypeError("labels must be a string or an iterable of strings")
      labels = list(labels)
    check.labels(labels, 'labels', single=False, unique=False)
    self._dash_labels: list[str] = labels

    check.integer(nb_digits, 'nb_digits', minimum=0)
    self._nb_digits: int = nb_digits

    check.choice(backend, 'backend', ('tkinter', 'pyqt'))
    self._backend: Literal['tkinter', 'pyqt'] = backend

    # Attributes related to PyQt6
    self._qt_labels: dict[str, QtWidgets.QLabel] = dict()
    self._qt_values: dict[str, QtWidgets.QLabel] = dict()

  def prepare(self) -> None:
    """Checks that there's at least one incoming
    :class:`~crappy.links.link.Link`, and starts the GUI.
    
    .. versionadded:: 1.5.7
    """

    if not self.inputs:
      raise IOError("No Link pointing towards the Dashboard Block !")
    if self.outputs:
      raise IOError("The Dashboard Block does not accept output Links!")

    self.log(logging.INFO, "Creating the dashboard window")
    if self._backend == 'tkinter':
      self._prepare_tkinter()
    else:
      self._prepare_pyqt()

  def loop(self) -> None:
    """Displays the latest requested values and services the interface, even
    when no new data is received.
    
    .. versionadded:: 1.5.7
    """

    data = self.recv_last_data(fill_missing=False)

    for label, value in data.items():
      if label not in self._dash_labels:
        continue
      if isinstance(value, str):
        text = value
      elif isinstance(value, numbers.Real):
        text = f'{float(value):.{self._nb_digits}f}'
      else:
        self.log(logging.WARNING, f"Don't know how to handle the received "
                                  f"value: {value}")
        continue

      self.log(logging.DEBUG, f"Displaying {text} for the label {label} "
                              f"on the dashboard")
      if self._backend == 'tkinter':
        try:
          assert self._dashboard is not None
          self._dashboard.tk_var[label].set(text)
        except tk.TclError:
          return
      else:
        self._qt_values[label].setText(text)

    if self._backend == 'tkinter':
      try:
        assert self._dashboard is not None
        self._dashboard.update()
      except tk.TclError:
        pass
    else:
      assert self._qt_app is not None
      self._qt_app.processEvents()

  def finish(self) -> None:
    """Closes the display, including after partial preparation.

    An existing Qt application is left available for its other windows.

    .. versionadded:: 1.5.7
    """

    manager = self._failure_manager

    # Guard cleanup steps with the FailureManager
    with manager.collect('Dashboard cleanup'):
      with manager.step('cleanup diagnostic'):
        self.log(logging.INFO, "Closing the GUI")

      # Destroy the Tk window
      if self._dashboard is not None:
        with manager.step('destroy Tk window'):
          try:
            self._dashboard.destroy()
          except tk.TclError:
            pass
          self._dashboard = None

      # Close the Qt window
      if self._qt_window is not None:
        self._qt_events_processed = False
        with manager.step('close Qt window'):
          if not self._qt_window.close():
            raise RuntimeError("The Dashboard Qt window refused to close")
          self._qt_window = None

      # Process the last Qt events
      if self._qt_app is not None and not self._qt_events_processed:
        with manager.step('process Qt close events'):
          self._qt_app.processEvents()
          self._qt_events_processed = True

  def _prepare_tkinter(self) -> None:
    """Creates the Tkinter window with label names and their latest values."""

    tk.resolve()
    from ._dashboard_tkinter import DashboardWindow

    self._dashboard = DashboardWindow(self._dash_labels)
    assert self._dashboard is not None
    self._dashboard.report_callback_exception = (
      self._report_tk_callback_exception)
    self._dashboard.update()

  def _prepare_pyqt(self) -> None:
    """Creates a fixed-size Qt window using the application's color palette.

    The two columns, font, and spacing match the Tkinter layout.
    """

    self._qt_app = self._get_application()
    self._qt_window = QtWidgets.QWidget()
    assert self._qt_window is not None
    self._qt_window.setWindowTitle('Dashboard')
    font = self._qt_window.font()
    font.setFamily('Courier')
    font.setPointSize(48)
    font.setBold(True)
    self._qt_window.setFont(font)

    layout = QtWidgets.QGridLayout(self._qt_window)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(30)
    layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

    self._qt_labels.clear()
    self._qt_values.clear()
    for row, label in enumerate(self._dash_labels):
      name = QtWidgets.QLabel(f'{label}:', self._qt_window)
      name.setTextFormat(QtCore.Qt.TextFormat.PlainText)
      value = QtWidgets.QLabel('', self._qt_window)
      value.setTextFormat(QtCore.Qt.TextFormat.PlainText)
      layout.addWidget(name, row, 0,
                       alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
      layout.addWidget(value, row, 1,
                       alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
      self._qt_labels[label] = name
      self._qt_values[label] = value

    self._qt_window.show()
    assert self._qt_app is not None
    self._qt_app.processEvents()

  def _get_application(self) -> QtWidgets.QApplication:
    """Reuses a Qt widget application or creates one for this Block.

    OpenCV's bundled Qt plugin path is excluded during application creation,
    then restored. The numeric locale is also preserved for tkinter windows.
    """

    app = QtCore.QCoreApplication.instance()
    if app is not None:
      if not isinstance(app, QtWidgets.QApplication):
        raise RuntimeError("A non-widget Qt application already exists")
      return app

    manager = self._failure_manager
    original = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')
    numeric_locale = None
    with manager.collect('Qt application setup'):
      with manager.step('create application'):
        numeric_locale = locale.setlocale(locale.LC_NUMERIC)
        if original is not None:
          paths = original.split(os.pathsep)
          filtered = [path for path in paths if
                      tuple(os.path.normpath(path).split(os.sep)[-3:]) !=
                      ('cv2', 'qt', 'plugins')]
          if filtered != paths:
            if filtered:
              os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = (
                  os.pathsep.join(filtered))
            else:
              os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        app = QtWidgets.QApplication([])
      with manager.step('restore numeric locale'):
        if numeric_locale is not None:
          locale.setlocale(locale.LC_NUMERIC, numeric_locale)
      with manager.step('restore Qt plugin path'):
        if original is not None:
          os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = original
    return app


def __getattr__(name: str) -> type:
  """Preserves the public Tk window alias without loading it for Qt users.

  .. versionadded:: 2.2.0
  """

  if name != 'DashboardWindow':
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

  from ._dashboard_tkinter import DashboardWindow

  globals()[name] = DashboardWindow
  return DashboardWindow


def __dir__() -> list[str]:
  """Includes the deferred Tk window class in attribute discovery.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
