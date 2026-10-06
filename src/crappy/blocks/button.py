# coding: utf-8

from __future__ import annotations
from time import time
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


class Button(Block):
  """This Block allows the user to send a signal to downstream Blocks upon
  clicking on a button in a Graphical User Interface.

  It sends an integer value, that starts from `0` and is incremented every time
  the user clicks on the button. The graphical interface uses PyQt6 by default,
  or :mod:`tkinter` when requested. Both backends display the current step
  above a button labeled ``'Next step'``. Closing the window does not stop the
  test, and periodic sending continues when ``spam`` is :obj:`True`.

  This Block is mostly useful for incorporating user feedback in a script, i.e.
  triggering actions based on an experimenter's decision. It can be handy for
  taking pictures at precise moments, or when an action should only begin after
  the experimenter has completed a task, for example.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *GUI* to *Button*
  .. versionchanged:: 2.1.0 the default backend is now PyQt, no longer Tkinter
  """

  def __init__(self,
               send_0: bool = False,
               label: str = 'step',
               time_label: str = 't(s)',
               backend: Literal['tkinter', 'pyqt'] = 'pyqt',
               freq: float | None = 50,
               spam: bool = False,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      send_0: If :obj:`True`, the value `0` will be sent automatically when
        starting the Block. Otherwise, `1` will be sent at the first click.
        Only relevant when ``spam`` is :obj:`False`.

        .. versionadded:: 1.5.10
      label: The label carrying the information on the number of clicks,
        default is ``'step'``.
      time_label: The label carrying the time information, default is
        ``'t(s)'``.

        .. versionadded:: 1.5.10
      backend: The library used for the graphical interface, either
        ``'pyqt'`` (the default) or ``'tkinter'``. The ``'pyqt'`` backend
        requires PyQt6. Both backends use the same layout and send the same
        signals.

        .. versionadded:: 2.1.0
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      spam: If :obj:`True`, sends the current step value at each loop,
        otherwise only sends it at each click.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionadded:: 2.0.0
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0
    """

    warn("\nThe default backend for the Button Block was changed from "
         "'tkinter' to 'pyqt' without prior notice.\nThis change was "
         "implemented nevertheless because it is part of a larger migration "
         "to PyQt\nSet backend='tkinter' to switch back to the "
         "previous behavior.\nInstall PyQt6 to use the new Button "
         "interface.\n", UserWarning, stacklevel=2)

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match send_0:
      case bool():
        self._send_0: bool = send_0
      case _:
        raise TypeError("send_0 mut be provided as a boolean")

    match label:
      case str() if label.strip():
        pass
      case str():
        raise ValueError("The label must be provided as a non-empty string")
      case _:
        raise TypeError("The label must be provided as a non-empty string")

    match time_label:
      case str() if time_label.strip():
        if time_label == label:
          raise ValueError("The time_label and the label must be different")
      case str():
        raise ValueError("the time_label must be provided as a non-empty "
                         "string")
      case _:
        raise TypeError("the time_label must be provided as a non-empty "
                        "string")

    match backend:
      case str() if label.strip() and backend in ('tkinter', 'pyqt'):
        self._backend: Literal['tkinter', 'pyqt'] = backend
      case str():
        raise ValueError("The backend must be either 'tkinter' or 'pyqt'")
      case _:
        raise TypeError("The backend must be either 'tkinter' or 'pyqt'")

    match spam:
      case bool():
        self._spam: bool = spam
      case _:
        raise TypeError("spam mut be provided as a boolean")

    self.labels = [time_label, label]

    # Keep the signal value independent of the graphical backend
    self._step: int = 0

    # Attributes related to tkinter
    self._root: tk.Tk | None = None
    self._text: tk.StringVar | None = None
    self._label: tk.Label | None = None
    self._button: tk.Button | None = None

    # Attributes related to PyQt6
    self._qt_app: QtWidgets.QApplication | None = None
    self._qt_window: QtWidgets.QWidget | None = None
    self._qt_label: QtWidgets.QLabel | None = None
    self._qt_button: QtWidgets.QPushButton | None = None
    self._callback_error: BaseException | None = None

  def prepare(self) -> None:
    """Creates the graphical interface and sets its layout and callbacks."""

    if not self.outputs:
      raise IOError("The Button Block has no output Link!")
    if self.inputs:
      raise IOError("The Button Block does not accept input Links!")

    self.log(logging.INFO, "Creating the GUI")

    if self._backend == 'tkinter':
      self._prepare_tkinter()
    else:
      self._prepare_pyqt()

  def begin(self) -> None:
    """Sends the value of the first step (`0`) if required."""

    if self._send_0 or self._spam:
      self.send([time() - self.t0, self._step])

  def loop(self) -> None:
    """Updates the interface, and sends the current step value if ``spam`` is
    :obj:`True`."""

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

    if self._spam:
      self.send([time() - self.t0, self._step])

  def finish(self) -> None:
    """Closes the interface window, including after partial preparation.

    An existing Qt application is left available for its other windows.
    """

    self.log(logging.INFO, "Closing the GUI")
    try:
      if getattr(self, '_root', None) is not None:
        assert self._root is not None
        self._root.destroy()
    except tk.TclError:
      pass

    if getattr(self, '_qt_window', None) is not None:
      assert self._qt_window is not None
      self._qt_window.close()
    if getattr(self, '_qt_app', None) is not None:
      assert self._qt_app is not None
      self._qt_app.processEvents()

  def _prepare_tkinter(self) -> None:
    """Creates the tkinter window with a counter above the step button."""

    self._root = tk.Tk()
    assert self._root is not None
    self._root.title("Button block")
    self._root.resizable(False, False)

    self._text = tk.StringVar(self._root,
                              value=f'{self.labels[1]}: {self._step}')
    assert self._text is not None
    self._label = tk.Label(self._root, textvariable=self._text)
    assert self._label is not None
    self._label.pack(padx=7, pady=7)

    self._button = tk.Button(self._root,
                             text='Next step',
                             command=self._next_step)
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
    self._qt_window.setWindowTitle("Button block")

    layout = QtWidgets.QVBoxLayout(self._qt_window)
    layout.setContentsMargins(7, 7, 7, 7)
    layout.setSpacing(14)
    layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

    self._qt_label = QtWidgets.QLabel(f'{self.labels[1]}: {self._step}',
                                      self._qt_window)
    assert self._qt_label is not None
    self._qt_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    layout.addWidget(self._qt_label,
                     alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    # Match the button's wider horizontal padding in the tkinter layout
    button_layout = QtWidgets.QHBoxLayout()
    button_layout.setContentsMargins(18, 0, 18, 0)
    self._qt_button = QtWidgets.QPushButton('Next step', self._qt_window)
    assert self._qt_button is not None
    self._qt_button.clicked.connect(self._next_step_pyqt)
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

  def _update_text(self) -> None:
    """Updates the displayed counter for the selected backend."""

    text = f'{self.labels[1]}: {self._step}'
    if self._backend == 'tkinter':
      assert self._text is not None
      self._text.set(text)
    else:
      assert self._qt_label is not None
      self._qt_label.setText(text)

  def _next_step_pyqt(self, _checked: bool = False) -> None:
    """Retains Qt callback failures for the Block's loop to raise.

    Letting an exception escape a Qt signal callback can abort the process.
    """

    try:
      self._next_step()
    except BaseException as error:
      self._callback_error = error
      assert self._qt_button is not None
      self._qt_button.setEnabled(False)

  def _next_step(self) -> None:
    """Increments the step counter and sends the corresponding signal."""

    self.log(logging.DEBUG, "Next step on the GUI")
    self._step += 1
    self._update_text()
    self.send([time() - self.t0, self._step])
