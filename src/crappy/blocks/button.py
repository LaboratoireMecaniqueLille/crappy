# coding: utf-8

from time import time
import logging
from typing import Optional

from .meta_block import Block
from .._global import OptionalModule

try:
  import tkinter as tk
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")


class Button(Block):
  """This Block allows the user to send a signal to downstream Blocks upon
  clicking on a button in a Graphical User Interface.

  It sends an integer value, that starts from `0` and is incremented every time
  the user clicks on the button. This Block relies on a :obj:`~tkinter.Tk`
  window for the graphical interface.

  This Block is mostly useful for incorporating user feedback in a script, i.e.
  triggering actions based on an experimenter's decision. It can be handy for
  taking pictures at precise moments, or when an action should only begin after
  the experimenter has completed a task, for example.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *GUI* to *Button*
  """

  def __init__(self,
               send_0: bool = False,
               label: str = 'step',
               time_label: str = 't(s)',
               freq: Optional[float] = 50,
               spam: bool = False,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
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

    self._root: Optional[tk.Tk] = None

    super().__init__()
    self.freq = freq
    self.labels = [time_label, label]
    self.display_freq = display_freq
    self.debug = debug

    self._spam = spam
    self._send_0 = send_0

    self._step = None
    self._text = None
    self._label = None
    self._button = None

  def prepare(self) -> None:
    """Creates the graphical interface and sets its layout and callbacks."""

    self.log(logging.INFO, "Creating the GUI")

    self._root = tk.Tk()
    self._root.title("Button block")
    self._root.resizable(False, False)

    self._step = tk.IntVar()
    self._step.trace_add('write', self._update_text)
    self._text = tk.StringVar(value=f'step: {self._step.get()}')

    self._label = tk.Label(self._root, textvariable=self._text)
    self._label.pack(padx=7, pady=7)

    self._button = tk.Button(self._root,
                             text='Next step',
                             command=self._next_step)
    self._button.pack(padx=25, pady=7)

    self._root.update()

  def begin(self) -> None:
    """Sends the value of the first step (`0`) if required."""

    if self._send_0:
      self.send([time() - self.t0, self._step.get()])

  def loop(self) -> None:
    """Updates the interface, and sends the current step value if ``spam`` is
    :obj:`True`. """

    try:
      self._root.update()
      self.log(logging.DEBUG, "GUI updated")
    except tk.TclError:
      return

    if self._spam:
      self.send([time() - self.t0, self._step.get()])

  def finish(self) -> None:
    """Closes the interface window."""

    self.log(logging.INFO, "closing the GUI")
    try:
      if self._root is not None:
        self._root.destroy()
    except tk.TclError:
      pass

  def _update_text(self, _, __, ___) -> None:
    """Simply updates the displayed text."""

    self._text.set(f'step: {self._step.get()}')

  def _next_step(self) -> None:
    """Increments the step counter and sends the corresponding signal."""

    self.log(logging.DEBUG, "Next step on the GUI")
    self._step.set(self._step.get() + 1)
    self.send([time() - self.t0, self._step.get()])
