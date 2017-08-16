# coding: utf-8

import serial
from threading import Thread
import tkinter as tk
from queue import Queue as Queue_threading, Empty
from time import time
from collections import OrderedDict
from multiprocessing import Process, Queue
from ast import literal_eval

from ..tool.GUI_Arduino.minitens import MinitensFrame
from ..tool.GUI_Arduino.arduino_basics import MonitorFrame, SubmitSerialFrame

from .inout import InOut
from .._global import CrappyStop
from os.path import exists

def collect_serial(arduino, queue):
  """Collect serial information, in a parallel way."""
  while True:
    queue.put(arduino.readline())


class ArduinoHandler(object):
  """
  This class creates every object (GUIs, Arduinos) and handles communication
  between them. inputs/ouputs of arduino and GUIs. The user doesn't interact
  directly with it, the Arduino IOBlock will create this handler.
  The ArduinoHandler lives on a separate process from the ArduinoIOBlock.
  """

  def __init__(self, *args):

    kwargs = args[0]  # Because one cannot pass multiple kwargs when creating
    #  a process...

    for key, value in kwargs.iteritems():
      setattr(self, key, value)
    self.arduino_ser = serial.Serial(port=self.port,
                                     baudrate=self.baudrate)

    self.collect_serial_queue = Queue_threading()  # To collect serial
    self.submit_serial_queue = Queue_threading()  # To send in serial

    # A thread that runs independently to collect serial port continuously.
    self.collect_serial_threaded = Thread(target=collect_serial,
                                          args=(self.arduino_ser,
                                                self.collect_serial_queue))
    self.collect_serial_threaded.daemon = True
    self.init_main_window()
    self.collect_serial_threaded.start()
    self.bool_loop = True

    self.main_loop()

  def init_main_window(self):
    """ Creates every frame specified by user, and creates links between proper
    objects. """
    self.root = tk.Tk()
    self.root.resizable(width=False, height=False)
    self.root.title("Arduino on crappy v1.3")
    self.root.protocol("WM_DELETE_WINDOW", self.close)

    if "monitor" in self.frames:
      title = "Arduino on port %s baudrate %s" % (self.port, self.baudrate)

      self.monitor_frame = MonitorFrame(self.root,
                                        width=int(self.width * 7 / 10),
                                        fontsize=self.fontsize,
                                        title=title)
      self.monitor_frame.pack()

    if "submit" in self.frames:
      self.submit_frame = SubmitSerialFrame(self.root,
                                            fontsize=self.fontsize,
                                            width=self.width,
                                            queue=self.submit_serial_queue)
      self.submit_frame.pack()

    if "minitens" in self.frames:
      # The minitens frame modifies  the dictionary that comes from the
      # arduino, and passes it to the crappy link.
      self.crappy_queue = Queue_threading()


      self.minitens_frame = MinitensFrame(self.root,
                                          queue=self.submit_serial_queue,
                                          width=self.width,
                                          fontsize=self.fontsize,
                                          crappy_queue=self.crappy_queue)

      self.root.config(menu=self.minitens_frame.menubar)
      self.minitens_frame.pack()

  def update_serial(self):
    """ Collect serial and writes in it (if applicable). Returns received
    information, or None if nothing received in 0.01 secs."""
    try:
      # Receiving from arduino
      serial_received = self.collect_serial_queue.get(block=True,
                                                      timeout=0.01)
    except Empty:
      # In case there is a queue timeout, to update GUI anyway
      serial_received = None
      self.root.update()
    try:
      # Sending to arduino
      serial_to_send = self.submit_serial_queue.get(block=False)
      self.arduino_ser.write(serial_to_send)
    except Empty:
      pass
    return serial_received

  def send_GUIs(self, serial_received):
    """
    send to every created GUI information received from arduino (if applicable.)
    """
    if "monitor" in self.frames:
      self.monitor_frame.update_widgets(serial_received)

    if "minitens" in self.frames:
      try:
        message = literal_eval(serial_received)
        self.minitens_frame.update_data(message)
      except (ValueError, SyntaxError, TypeError):
        pass

  def send_crappy(self, serial_received):
    """
    Depending on which GUI is created, multiple cases can occur.
    - If monitor and/or submit GUI is created, the arduino string returned
    must be evaluated as a dict.
    - If minitens GUI is created, it returns a dict.
    """
    if isinstance(serial_received, dict):
      self.queue_process.put(serial_received)
    elif isinstance(serial_received, str):

      try:
        message = literal_eval(serial_received)
        self.queue_process.put(message)
      except (ValueError, SyntaxError, TypeError) as e:
        print("[Arduino] %s: Skipping data" % e)

  def main_loop(self):
    """
    Update GUI, inputs and outputs.
    """
    while True and self.bool_loop:
      serial_received = self.update_serial()
      if serial_received:
        self.send_GUIs(serial_received)

      if "minitens" in self.frames:
        try:
          new_data = self.crappy_queue.get(block=False)
          self.send_crappy(new_data)
        except Empty:
          pass
      else:
        self.send_crappy(serial_received)

      self.root.update()
    # Executed if user closes the window. For proper CrappyStopping.
    self.root.destroy()
    self.queue_process.put("STOP")

  def close(self):
    """
    Exit main loop.
    """
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
      frames: Which frames to show. Avaiable:
      - monitor,
      - submit,
      - minitens.

    """

    if not kwargs.pop("port", None):
      # Tries to open the 5 first ttyACM's, that should be enough.
      for i in xrange(5):  #
        if exists('/dev/ttyACM' + str(i)):
          self.port = '/dev/ttyACM' + str(i)
          break

    for arg, default in [("baudrate", 9600),
                         ("labels", None),
                         ("frames", ["monitor", "submit"]),
                         ("width", 100),
                         ("fontsize", 11)]:

      setattr(self, arg, kwargs.pop(arg, default))

    assert not kwargs, "[Arduino]: unknown kwarg(s):" + str(kwargs)
    self.queue_get_data = Queue()

  def open(self):
    """
    Opens ArduinoHandler.
    """
    args_handler = {"port": self.port,
                    "baudrate": self.baudrate,
                    "queue_process": self.queue_get_data,
                    "width": self.width,
                    "fontsize": self.fontsize,
                    "frames": self.frames,
                    "labels": self.labels}

    self.arduino_handler = Process(target=ArduinoHandler, args=(args_handler,))
    self.handler_t0 = time()
    self.arduino_handler.start()

  def get_data(self, mock=None):
    """
    Gets data from arduinoHandler, or the minitens GUI
    """
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
