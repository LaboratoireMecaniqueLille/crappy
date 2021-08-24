# coding: utf-8

"""
A window that replicates the behavior of the serial monitor of the official
Arduino IDE. One can see the serial port entrances, and can write in it.
"""

from queue import Empty

from ..._global import OptionalModule

try:
  import tkinter as tk
  from tkinter import font as tk_font
except (ModuleNotFoundError, ImportError):
  tk = OptionalModule("tkinter")
  tk_font = OptionalModule("tkinter")


class MonitorFrame(tk.Frame):
  """A frame that displays everything entering the serial port.

  Everything is handled by ArduinoHandler, so you don't get to modify these
  values.

  Args:
    arduino: serial.Serial of arduino board.
    width: size of the text frame
    title: the title of the frame.
    fontsize: size of font inside the text frame.
  """

  def __init__(self, parent, **kwargs):
    tk.Frame.__init__(self, parent)
    # self.grid()
    self.total_width = kwargs.get('width', 100 * 5 / 10)
    self.arduino = kwargs.get("arduino")
    self.queue = kwargs.get("queue")

    # A checkbox to enable or disable displaying of information.
    self.enabled_checkbox = tk.IntVar()
    self.enabled_checkbox.set(1)

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):
    """Widgets shown:

    - The frame's title,
    - The checkbutton to enable/disable displaying,
    - The textbox.
    """

    self.top_frame = tk.Frame(self)
    tk.Label(self.top_frame,
             text=kwargs.get('title', '')).grid(row=0, column=0)

    tk.Checkbutton(self.top_frame,
                   variable=self.enabled_checkbox,
                   text="Display?").grid(row=0, column=1)

    self.serial_monitor = tk.Text(self,
                                  relief="sunken",
                                  height=int(self.total_width / 10),
                                  width=int(self.total_width),
                                  font=tk_font.Font(size=kwargs.get("fontsize",
                                                                    13)))

    self.top_frame.grid(row=0)
    self.serial_monitor.grid(row=1)

  def update_widgets(self, arg):
    if self.enabled_checkbox.get():
      self.serial_monitor.insert("0.0", arg)  # To insert at the top


class SubmitSerialFrame(tk.Frame):
  """Frame that permits to submit to the serial port of arduino.

  Args:
    width: width of the frame.
    fontsize: self-explanatory.
  """

  def __init__(self, parent, **kwargs):
    tk.Frame.__init__(self, parent)
    # self.grid()
    self.total_width = kwargs.get("width", 100)
    self.queue = kwargs.get("queue")

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):
    """Widgets shown:

    - an Input text, to enter what to write on serial port.
    - a submit label, to show previous command submitted.
    - a submit button.
    """

    self.input_txt = tk.Entry(self,
                              width=self.total_width * 5 / 10,
                              font=tk_font.Font(
                                size=kwargs.get("fontsize", 13)))
    self.submit_label = tk.Label(self, text='',
                                 width=1,
                                 font=tk_font.Font(
                                   size=kwargs.get("fontsize", 13)))
    self.submit_button = tk.Button(self,
                                   text='Submit',
                                   command=self.update_widgets,
                                   width=int(self.total_width * 0.5 / 10),
                                   font=tk_font.Font(
                                     size=kwargs.get("fontsize", 13)))

    self.input_txt.bind('<Return>', self.update_widgets)
    self.input_txt.bind('<KP_Enter>', self.update_widgets)

    # Positioning
    self.input_txt.grid(row=0, column=0, sticky=tk.W)
    self.submit_label.grid(row=0, column=1)
    self.submit_button.grid(row=0, column=2, sticky=tk.E)

  def update_widgets(self):
    try:
      message = self.queue.get(block=False)
    except Empty:
      message = self.input_txt.get()
    self.input_txt.delete(0, 'end')

    if len(message) > int(self.total_width / 4):
      # A fancy feature to resize the entry's length.
      self.input_txt.configure(width=int(self.total_width * 5 / 10 - len(
        message)))
    else:
      self.input_txt.configure(width=int(self.total_width * 5 / 10))
    self.submit_label.configure(width=len(message))
    self.submit_label.configure(text=message)
    self.queue.put(message)
