# coding: utf-8

import logging
from typing import Optional

from .meta_block import Block
from .._global import OptionalModule

try:
  import tkinter as tk
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")


class StopButton(Block):
  """This Block allows the user to stop the current Crapy script by clicking on
  a button in a GUI.

  Along with the :class:`~crappy.blocks.StopBlock`, it allows to stop a test in
  a clean way without resorting to CTRL+C.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
    """

    self._root: Optional[tk.Tk] = None

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug
    self.pausable = False

    self._label = None
    self._button = None

  def prepare(self) -> None:
    """Creates the graphical interface and sets its layout and callbacks."""

    self.log(logging.INFO, "Creating the GUI")

    self._root = tk.Tk()
    self._root.title("Stop Button Block")
    self._root.resizable(False, False)

    self._label = tk.Label(self._root, text="Click button to stop test")
    self._label.pack(padx=7, pady=7)

    self._button = tk.Button(self._root,
                             text='STOP',
                             command=self._clicked)
    self._button.pack(padx=25, pady=7)

    self._root.update()

  def loop(self) -> None:
    """Updates the interface at each loop."""

    try:
      self._root.update()
      self.log(logging.DEBUG, "GUI updated")
    except tk.TclError:
      return

  def finish(self) -> None:
    """Closes the interface window."""

    try:
      if self._root is not None:
        self.log(logging.INFO, "Closing the GUI")
        self._root.destroy()
    except tk.TclError:
      pass

  def _clicked(self) -> None:
    """When the stop button is clicked, stops the test."""

    self.log(logging.DEBUG, "Button clicked in the GUI")
    self.log(logging.WARNING, "Stop button clicked, stopping the script !")
    self.stop()
