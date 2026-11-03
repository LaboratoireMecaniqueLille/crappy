# coding: utf-8

"""Tk-only Dashboard window, imported only when its backend is requested."""

import tkinter as tk


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
