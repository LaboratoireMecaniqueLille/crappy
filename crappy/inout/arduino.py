# coding: utf-8

import serial
from threading import Thread
import Tkinter as tk
import tkFont
from Queue import Queue as Queue_threading, Empty
from time import time
from collections import OrderedDict
from multiprocessing import Process, Queue
from ast import literal_eval
from ..tool.minitens import MinitensFrame
from .inout import InOut
from .._global import CrappyStop
from os.path import exists

class MonitorFrame(tk.Frame):
  """
  A frame that displays everything entering the serial port.
  Everything is handled by ArduinoHandler, so you don't get to modify these
  values.
  args:
    arduino: serial.Serial of arduino board.
    width: size of the text frame
    title: the title of the frame.
    fontsize: size of font inside the text frame.
  """

  def __init__(self, parent, **kwargs):
    tk.Frame.__init__(self, parent)
    self.grid()
    self.total_width = kwargs.get('width', 100 * 5 / 10)
    self.arduino = kwargs.get("arduino")
    self.queue = kwargs.get("queue")

    # A checkbox to enable or disable displaying of information.
    self.enabled_checkbox = tk.IntVar()
    self.enabled_checkbox.set(1)

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):
    """
    Widgets shown:
    - The frame's title,
    - The checkbutton to enable/disable displaying,
    - The textbox.
    
    """
    self.top_frame = tk.Frame(self)
    tk.Label(self.top_frame, text=kwargs.get('title', '')).grid(row=0, column=0)

    tk.Checkbutton(self.top_frame,
                   variable=self.enabled_checkbox,
                   text="Display?").grid(row=0, column=1)

    self.serial_monitor = tk.Text(self,
                                  relief="sunken",
                                  height=int(self.total_width / 10),
                                  width=int(self.total_width),
                                  font=tkFont.Font(size=kwargs.get("fontsize",
                                                                   13)))

    self.top_frame.grid(row=0)
    self.serial_monitor.grid(row=1)

  def update_widgets(self, arg):
    if self.enabled_checkbox.get():
      self.serial_monitor.insert("0.0", arg)  # To insert at the top


class SubmitSerialFrame(tk.Frame):
  """
  Frame that permits to submit to the serial port of arduino.

  args:
    width: width of the frame.
    fontsize: self-explanatory.
  """

  def __init__(self, parent, **kwargs):
    tk.Frame.__init__(self, parent)
    self.grid()
    self.total_width = kwargs.get("width", 100)
    self.queue = kwargs.get("queue")

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):
    """
    Widgets shown:
    - an Input text, to enter what to write on serial port.
    - a submit label, to show previous command submitted.
    - a submit button.
    """

    self.input_txt = tk.Entry(self,
                              width=self.total_width * 5 / 10,
                              font=tkFont.Font(size=kwargs.get("fontsize", 13)))
    self.submit_label = tk.Label(self, text='',
                                 width=1,
                                 font=tkFont.Font(
                                   size=kwargs.get("fontsize", 13)))
    self.submit_button = tk.Button(self,
                                   text='Submit',
                                   command=self.update_widgets,
                                   width=int(self.total_width * 0.5 / 10),
                                   font=tkFont.Font(
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


class ArduinoHandler(object):
  """
  Main ArduinoHandler, that creates every frame and handles in and outs from/to
  every frame and arduino.

  """
  def __init__(self, port, baudrate, queue_process, width, fontsize, frames):

    def collect_serial(arduino, queue):
      """Collect serial information, in a parallel way."""
      while True:
        queue.put(arduino.readline())

    self.port = port
    self.baudrate = baudrate
    self.queue_process = queue_process
    self.width = width
    self.fontsize = fontsize
    self.frames = frames

    self.arduino_ser = serial.Serial(port=self.port,
                                     baudrate=self.baudrate)

    self.collect_serial_queue = Queue_threading()  # To collect serial
    # information
    self.submit_serial_queue = Queue_threading()  # To collect user commands
    # and send it to serial

    self.collect_serial_threaded = Thread(target=collect_serial,
                                          args=(self.arduino_ser,
                                                self.collect_serial_queue))
    self.collect_serial_threaded.daemon = True
    self.init_main_window()
    self.collect_serial_threaded.start()
    self.bool_loop = True
    self.main_loop()

  def init_main_window(self):
    """
    Method to create and place widgets inside the main window.
    """
    self.root = tk.Tk()
    self.root.resizable(width=False, height=False)
    self.root.title("Arduino on crappy v1.2")
    self.root.protocol("WM_DELETE_WINDOW", self.close)
    if "monitor" in self.frames:
      self.monitor_frame = MonitorFrame(self.root,
                                        width=int(self.width * 7 / 10),
                                        fontsize=self.fontsize,
                                        title="Arduino on port %s "
                                              "baudrate %s" % (self.port,
                                                               self.baudrate))
      self.monitor_frame.grid()
    if "submit" in self.frames:
      self.submit_frame = SubmitSerialFrame(self.root,
                                            fontsize=self.fontsize,
                                            width=self.width,
                                            queue=self.submit_serial_queue)
      self.submit_frame.grid()
    if "minitens" in self.frames:
      self.minitens_frame = MinitensFrame(self.root,
                                          queue=self.submit_serial_queue,
                                          width=self.width,
                                          fontsize=self.fontsize)
      self.root.config(menu=self.minitens_frame.menubar)
      self.minitens_frame.grid()

  def main_loop(self):
    """
    Main method to update the GUI, collect and transmit information.
    """
    while True and self.bool_loop:
      try:
        serial_received = self.collect_serial_queue.get(block=True,
                                                        timeout=0.01)
      except Empty:
        serial_received = None
        self.root.update()  # In case there is a queue timeout
      try:
        serial_to_send = self.submit_serial_queue.get(block=False)
        self.arduino_ser.write(serial_to_send)
      except Empty:
        pass

      if "monitor" in self.frames and serial_received:
        self.monitor_frame.update_widgets(serial_received)

      if "minitens" in self.frames and serial_received:
        try:
          message = literal_eval(serial_received)
          self.minitens_frame.update_data(message)
        except (ValueError, SyntaxError, TypeError):
          pass

      if serial_received:
        try:
          message = literal_eval(serial_received)
          self.queue_process.put(message)  # Message is sent to the crappy
          # process
        except (ValueError, SyntaxError, TypeError):
          pass
      self.root.update()
    self.root.destroy()
    self.queue_process.put("STOP")

  def close(self):
    self.bool_loop = False


class Arduino(InOut):
  def __init__(self, **kwargs):
    """
    Main class used to interface Arduino, its GUI and crappy. For
    reusability, make sure the program inside the arduino sends to the serial
    port a python dictionary formated string.

    Args:
      port: serial port of the arduino.
      baudrate: baudrate defined inside the arduino program.
      width: width of the GUI.
      fontsize: size of the font inside the GUI.
    """
    for arg, default in [('port', None),
                         ("baudrate", 9600),
                         ("labels", None),
                         ("frames", ["monitor", "submit"]),
                         ("width", 100),
                         ("fontsize", 11)]:
      setattr(self, arg, kwargs.pop(arg, default))

    if not self.port:
      for i in xrange(5):  #
        if exists('/dev/ttyACM' + str(i)):
          self.port = '/dev/ttyACM' + str(i)
          break

    assert not kwargs, "arduino: unknown kwarg(s):" + str(kwargs)

  def open(self):
    self.queue_get_data = Queue()
    self.arduino_handler = Process(target=ArduinoHandler,
                                   args=(self.port,
                                         self.baudrate,
                                         self.queue_get_data,
                                         self.width,
                                         self.fontsize,
                                         self.frames))
    self.handler_t0 = time()
    self.arduino_handler.start()

  def get_data(self, mock=None):

    retrieved_from_arduino = self.queue_get_data.get()
    if retrieved_from_arduino == "STOP":
      raise CrappyStop
    if self.labels:
      ordered = OrderedDict()
      ordered["time(sec)"] = 0.
      for key in self.labels:
        ordered[key] = retrieved_from_arduino[key]
      return ordered
    else:
      return retrieved_from_arduino

  def close(self):
    self.arduino_handler.terminate()
