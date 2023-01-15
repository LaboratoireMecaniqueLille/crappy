# coding: utf-8

from typing import List, Optional
import tkinter as tk
import logging

from .meta_block import Block


class Dashboard_window(tk.Tk):
  """The GUI for displaying the label values."""

  def __init__(self, labels: List[str]) -> None:
    """Initializes the GUI and sets the layout."""

    super().__init__()
    self.title('Dashboard')
    self.resizable(False, False)

    self._labels = labels

    # Attributes storing the tkinter objects
    self._tk_labels = {}
    self._tk_values = {}
    self.tk_var = {}

    # Setting the GUI
    self._set_variables()
    self._set_layout()

  def _set_variables(self) -> None:
    """Attributes one StringVar per label."""

    for label in self._labels:
      self.tk_var[label] = tk.StringVar(value='')

  def _set_layout(self) -> None:
    """Creates the Labels and places them on the GUI."""

    for row, label in enumerate(self._labels):
      # The name of the labels on the left
      self._tk_labels[label] = tk.Label(self, text=label, borderwidth=15,
                                        font=("Courier bold", 48))
      self._tk_labels[label].grid(row=row, column=0)
      # Their values on the right
      self._tk_values[label] = tk.Label(self, borderwidth=15,
                                        textvariable=self.tk_var[label],
                                        font=("Courier bold", 48))
      self._tk_values[label].grid(row=row, column=1)


class Dashboard(Block):
  """The Dashboard receives data from a :ref:`Link`, and displays it on a new
  popped window.

  It can only display data coming from one block.
  """

  def __init__(self,
               labels: List[str],
               nb_digits: int = 2,
               verbose: bool = False,
               freq: float = 30,
               debug: bool = False) -> None:
    """Sets the args and initializes parent class.

    Args:
      labels: Only the data from these labels will be displayed on the window.
      nb_digits: Number of decimals to show.
      verbose: If :obj:`True`, displays the looping frequency of the block.
      freq: If set, the block will try to loop at this frequency.
    """

    self._dashboard: Optional[Dashboard_window] = None

    super().__init__()
    self.verbose = verbose
    self.freq = freq
    self.log_level = logging.DEBUG if debug else logging.INFO

    self._labels = labels
    self._nb_digits = nb_digits

  def prepare(self) -> None:
    """Checks that there's only one incoming link, and starts the GUI."""

    if len(self.inputs) == 0:
      raise IOError("No link pointing towards the Dashboard block !")
    elif len(self.inputs) > 1:
      raise IOError("Too many links pointing towards the Dashboard block !")

    self.log(logging.INFO, "Creating the dashboard window")
    self._dashboard = Dashboard_window(self._labels)
    self._dashboard.update()

  def loop(self) -> None:
    """Receives the data from the incoming link and displays it."""

    data = self.recv_last_data(fill_missing=False)

    for label, value in data.items():
      # Only displays the required labels
      if label in self._labels:
        # Possibility to display str values carried by the links
        if isinstance(value, str):
          self.log(logging.DEBUG, f"Displaying {value} for the label {label} "
                                  f"on the dashboard")
          self._dashboard.tk_var[label].set(value)
        elif isinstance(value, int) or isinstance(value, float):
          self.log(logging.DEBUG, f"Displaying {value:.{self._nb_digits}f} for"
                                  f" the label {label} on the dashboard")
          self._dashboard.tk_var[label].set(f'{value:.{self._nb_digits}f}')

    # In case the GUI has been destroyed, don't raise an error
    try:
      self._dashboard.update()
    except tk.TclError:
      pass

  def finish(self) -> None:
    """"""

    # In case the GUI has been destroyed, don't raise an error
    try:
      if self._dashboard is not None:
        self.log(logging.INFO, "Closing the dashboard window")
        self._dashboard.destroy()
    except tk.TclError:
      pass
