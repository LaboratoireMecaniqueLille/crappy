# coding: utf-8

from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import logging
import locale
import os
import tkinter as tk
from warnings import warn

from .meta_block import Block
from .._global import OptionalModule

QtCore = OptionalModule('PyQt6.QtCore', lazy_import=True)
QtWidgets = OptionalModule('PyQt6.QtWidgets', lazy_import=True)

if TYPE_CHECKING:
  from PyQt6 import QtWidgets, QtCore


class StopButton(Block):
  """This Block allows the user to stop the current Crappy script by clicking
  on a button in a GUI.

  Along with the :class:`~crappy.blocks.StopBlock`, it allows to stop a test in
  a clean way without resorting to CTRL+C.

  The graphical interface uses PyQt6 by default, or :mod:`tkinter` when
  requested. Both backends display the message ``'Click button to stop test'``
  above a button labeled ``'STOP'``. This Block remains active when the test
  is paused. Closing the window does not stop the test.

  This Block does not require or accept input or output Links.

  .. versionadded:: 2.0.0
  .. versionchanged:: 2.1.0 the default backend is now PyQt, no longer Tkinter
  """

  def __init__(self,
               backend: Literal['tkinter', 'pyqt'] = 'pyqt',
               freq: float | None = 50,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      backend: The library used for the graphical interface, either
        ``'pyqt'`` (the default) or ``'tkinter'``. The ``'pyqt'`` backend
        requires PyQt6. Both backends use the same layout and request a clean
        stop when the button is clicked.

        .. versionadded:: 2.1.0
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
    """

    warn("\nThe default backend for the StopButton Block was changed from "
         "'tkinter' to 'pyqt' without prior notice.\nThis change was "
         "implemented nevertheless because it is part of a larger migration "
         "to PyQt\nSet backend='tkinter' to switch back to the "
         "previous behavior.\nInstall PyQt6 to use the new StopButton "
         "interface.\n", UserWarning, stacklevel=2)

    self._root: tk.Tk | None = None
    self._qt_app: QtWidgets.QApplication | None = None
    self._qt_window: QtWidgets.QWidget | None = None
    self._qt_events_processed: bool = False

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug
    self.pausable = False

    match backend:
      case str() if backend in ('tkinter', 'pyqt'):
        self._backend: Literal['tkinter', 'pyqt'] = backend
      case str():
        raise ValueError("The backend must be either 'tkinter' or 'pyqt'")
      case _:
        raise TypeError("The backend must be either 'tkinter' or 'pyqt'")

    # Attributes related to tkinter
    self._label: tk.Label | None = None
    self._button: tk.Button | None = None

    # Attributes related to PyQt6
    self._qt_label: QtWidgets.QLabel | None = None
    self._qt_button: QtWidgets.QPushButton | None = None
    self._callback_error: BaseException | None = None

  def prepare(self) -> None:
    """Creates the graphical interface and sets its layout and callbacks."""

    if self.inputs:
      raise IOError("The StopButton Block does not accept input Links!")
    if self.outputs:
      raise IOError("The StopButton Block does not accept output Links!")

    self.log(logging.INFO, "Creating the GUI")

    if self._backend == 'tkinter':
      self._prepare_tkinter()
    else:
      self._prepare_pyqt()

  def loop(self) -> None:
    """Services the interface and raises any retained Qt callback failure."""

    if self._backend == 'tkinter':
      try:
        assert self._root is not None
        self._root.update()
      except tk.TclError:
        return
    else:
      assert self._qt_app is not None
      self._qt_app.processEvents()
      if self._callback_error is not None:
        raise self._callback_error

    self.log(logging.DEBUG, "GUI updated")

  def finish(self) -> None:
    """Closes the interface window, including after partial preparation.

    An existing Qt application is left available for its other windows.
    """

    failures: list[Exception | KeyboardInterrupt] = list()
    self.log(logging.INFO, "Closing the GUI")

    # Destroy the Tk window
    if self._root is not None:
      try:
        self._root.destroy()
      except tk.TclError:
        # Tk may already have destroyed the window
        self._root = None
      except (Exception, KeyboardInterrupt) as error:
        error.add_note("StopButton cleanup step: destroy Tk window")
        failures.append(error)
      else:
        self._root = None

    # Close the Qt Window
    if self._qt_window is not None:
      self._qt_events_processed = False
      try:
        if not self._qt_window.close():
          raise RuntimeError("The StopButton Qt window refused to close")
      except (Exception, KeyboardInterrupt) as error:
        error.add_note("StopButton cleanup step: close Qt window")
        failures.append(error)
      else:
        self._qt_window = None

    # Process the last Qt events
    if self._qt_app is not None and not self._qt_events_processed:
      try:
        self._qt_app.processEvents()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note("StopButton cleanup step: process Qt close events")
        failures.append(error)
      else:
        self._qt_events_processed = True

    # If there's only one Exception, raise it
    if len(failures) == 1:
      raise failures[0]
    # Handle the case when a KeyboardInterrupt is among the Exceptions
    elif any(isinstance(error, KeyboardInterrupt) for error in failures):
      for index, error in enumerate(failures):
        if isinstance(error, KeyboardInterrupt):
          others: list[BaseException] = failures[:index] + failures[index + 1:]
          if error.__cause__ is not None:
            others.insert(0, error.__cause__)
          raise error from BaseExceptionGroup("Other StopButton cleanup "
                                              "failures", others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("StopButton cleanup failures", failures)

  def _prepare_tkinter(self) -> None:
    """Creates the Tkinter window with a message above the stop button."""

    self._root = tk.Tk()
    assert self._root is not None
    self._root.title("Stop Button Block")
    self._root.resizable(False, False)

    self._label = tk.Label(self._root, text="Click button to stop test")
    assert self._label is not None
    self._label.pack(padx=7, pady=7)

    self._button = tk.Button(self._root,
                             text='STOP',
                             command=self._clicked)
    assert self._button is not None
    self._button.pack(padx=25, pady=7)

    self._root.update()

  def _prepare_pyqt(self) -> None:
    """Creates a fixed-size Qt window using the application's color palette.

    The label and button have the same order and spacing as in tkinter.
    """

    self._qt_app = self._get_application()
    self._qt_window = QtWidgets.QWidget()
    assert self._qt_window is not None
    self._qt_window.setWindowTitle("Stop Button Block")

    layout = QtWidgets.QVBoxLayout(self._qt_window)
    layout.setContentsMargins(7, 7, 7, 7)
    layout.setSpacing(14)
    layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

    self._qt_label = QtWidgets.QLabel("Click button to stop test",
                                      self._qt_window)
    assert self._qt_label is not None
    self._qt_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    layout.addWidget(self._qt_label,
                     alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    # Match the button's wider horizontal padding in the tkinter layout
    button_layout = QtWidgets.QHBoxLayout()
    button_layout.setContentsMargins(18, 0, 18, 0)
    self._qt_button = QtWidgets.QPushButton('STOP', self._qt_window)
    assert self._qt_button is not None
    self._qt_button.clicked.connect(self._clicked_pyqt)
    button_layout.addWidget(self._qt_button,
                            alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    layout.addLayout(button_layout)

    self._qt_window.show()
    assert self._qt_app is not None
    self._qt_app.processEvents()
    if self._callback_error is not None:
      raise self._callback_error

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

  def _clicked_pyqt(self, _checked: bool = False) -> None:
    """Retains Qt callback failures for the Block's loop to raise.

    Letting an exception escape a Qt signal callback can abort the process.
    """

    try:
      self._clicked()
    except BaseException as error:
      self._callback_error = error
      assert self._qt_button is not None
      self._qt_button.setEnabled(False)

  def _clicked(self) -> None:
    """When the stop button is clicked, stops the test."""

    self.log(logging.DEBUG, "Button clicked in the GUI")
    self.log(logging.WARNING, "Stop button clicked, stopping the script !")
    self.stop()
