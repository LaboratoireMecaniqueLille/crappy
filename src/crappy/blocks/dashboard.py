# coding: utf-8

from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
import tkinter as tk
import logging
import locale
import numbers
import os
from warnings import warn

from .meta_block import Block
from .._global import OptionalModule

QtCore = OptionalModule('PyQt6.QtCore', lazy_import=True)
QtWidgets = OptionalModule('PyQt6.QtWidgets', lazy_import=True)

if TYPE_CHECKING:
  from PyQt6 import QtWidgets, QtCore


class DashboardWindow(tk.Tk):
  """The Tkinter GUI for displaying the label values.
  
  .. versionadded:: 1.5.7
  .. versionchanged:: 2.0.0
     renamed from *Dashboard_window* to *DashboardWindow*
  """

  def __init__(self, labels: list[str]) -> None:
    """Initializes the GUI and sets the layout."""

    super().__init__()
    self.title('Dashboard')
    self.resizable(False, False)

    self._labels: list[str] = labels

    # Attributes storing the tkinter objects
    self._tk_labels: dict[str, tk.Label] = dict()
    self._tk_values: dict[str, tk.Label] = dict()
    self.tk_var: dict[str, tk.StringVar] = dict()

    # Setting the GUI
    self._set_variables()
    self._set_layout()

  def _set_variables(self) -> None:
    """Attributes one StringVar per label."""

    for label in self._labels:
      self.tk_var[label] = tk.StringVar(self, value='')

  def _set_layout(self) -> None:
    """Creates the Labels and places them on the GUI."""

    for row, label in enumerate(self._labels):
      # The name of the labels on the left
      self._tk_labels[label] = tk.Label(self, text=f'{label}:', borderwidth=15,
                                        font=("Courier bold", 48))
      self._tk_labels[label].grid(row=row, column=0)
      # Their values on the right
      self._tk_values[label] = tk.Label(self, borderwidth=15,
                                        textvariable=self.tk_var[label],
                                        font=("Courier bold", 48))
      self._tk_values[label].grid(row=row, column=1)


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

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match labels:
      case str() if labels.strip():
        self._dash_labels: list[str] = [labels]
      case str():
        raise ValueError("labels must contain non-empty strings")
      case Iterable() if not isinstance(labels, (bytes, bytearray)):
        self._dash_labels: list[str] = list(labels)
        if not self._dash_labels:
          raise ValueError("At least one label must be provided")
        for label in self._dash_labels:
          match label:
            case str() if label.strip():
              pass
            case str():
              raise ValueError("labels must contain non-empty strings")
            case _:
              raise TypeError("labels must contain non-empty strings")
      case _:
        raise TypeError("labels must be a string or an iterable of strings")

    match nb_digits:
      case bool():
        raise TypeError("nb_digits must be a non-negative integer")
      case int() if nb_digits >= 0:
        self._nb_digits: int = nb_digits
      case int():
        raise ValueError("nb_digits must be a non-negative integer")
      case _:
        raise TypeError("nb_digits must be a non-negative integer")

    match backend:
      case str() if backend in ('tkinter', 'pyqt'):
        self._backend: Literal['tkinter', 'pyqt'] = backend
      case str():
        raise ValueError("The backend must be either 'tkinter' or 'pyqt'")
      case _:
        raise TypeError("The backend must be either 'tkinter' or 'pyqt'")

    # Attributes related to tkinter
    self._dashboard: DashboardWindow | None = None

    # Attributes related to PyQt6
    self._qt_app: QtWidgets.QApplication | None = None
    self._qt_window: QtWidgets.QWidget | None = None
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

    self.log(logging.INFO, "Closing the dashboard window")
    try:
      if getattr(self, '_dashboard', None) is not None:
        assert self._dashboard is not None
        self._dashboard.destroy()
    except tk.TclError:
      pass

    if getattr(self, '_qt_window', None) is not None:
      assert self._qt_window is not None
      self._qt_window.close()
    if getattr(self, '_qt_app', None) is not None:
      assert self._qt_app is not None
      self._qt_app.processEvents()

  def _prepare_tkinter(self) -> None:
    """Creates the Tkinter window with label names and their latest values."""

    self._dashboard = DashboardWindow(self._dash_labels)
    assert self._dashboard is not None
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

  @staticmethod
  def _get_application() -> QtWidgets.QApplication:
    """Reuses a Qt widget application or creates one for this Block.

    OpenCV's bundled Qt plugin path is excluded during application creation,
    then restored. The numeric locale is also preserved for tkinter windows.
    """

    app = QtCore.QCoreApplication.instance()
    if app is not None:
      if not isinstance(app, QtWidgets.QApplication):
        raise RuntimeError("A non-widget Qt application already exists")
      return app

    original = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH')
    if original is not None:
      paths = original.split(os.pathsep)
      filtered = [path for path in paths if
                  tuple(os.path.normpath(path).split(os.sep)[-3:]) !=
                  ('cv2', 'qt', 'plugins')]
      if filtered != paths:
        if filtered:
          os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.pathsep.join(filtered)
        else:
          os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

    numeric_locale = locale.setlocale(locale.LC_NUMERIC)
    try:
      return QtWidgets.QApplication([])
    finally:
      locale.setlocale(locale.LC_NUMERIC, numeric_locale)
      if original is not None:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = original
