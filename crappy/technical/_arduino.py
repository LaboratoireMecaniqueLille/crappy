import serial
from threading import Thread
import Tkinter as tk
import tkFont
from Queue import Queue as Queue_threading, Empty
from time import sleep, time
from collections import OrderedDict
from multiprocessing import Process, Queue
from ast import literal_eval


class MonitorFrame(tk.Frame):
  def __init__(self, parent, *args, **kwargs):
    """
    A frame that displays everything enters the serial port.
    Args:
      arduino: serial.Serial of arduino board.
      width: size of the text frame
      title: the title of the frame.
      fontsize: size of font inside the text frame.
    """
    tk.Frame.__init__(self, parent)
    self.grid()
    self.total_width = kwargs.get('width', 100 * 5 / 10)
    self.arduino = kwargs.get("arduino")
    self.queue = kwargs.get("queue")
    self.enabled_checkbox = tk.IntVar()
    self.enabled_checkbox.set(1)

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):
    """
    Widgets shown : the title with option
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

  def update_widgets(self, *args):
    if self.enabled_checkbox.get():
      self.serial_monitor.insert("0.0", args[0])  # To insert at the top
    else:
      pass


class SubmitSerialFrame(tk.Frame):
  def __init__(self, parent, *args, **kwargs):
    """
    Frame that permits to submit to the serial port of arduino.
    Args:
      width: width of the frame.
      fontsize: self-explanatory.
    """
    tk.Frame.__init__(self, parent)
    self.grid()
    self.total_width = kwargs.get("width", 100)
    self.queue = kwargs.get("queue")

    self.create_widgets(**kwargs)

  def create_widgets(self, **kwargs):

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

  def update_widgets(self, *args):
    try:
      message = self.queue.get(block=False)
    except Empty:
      message = self.input_txt.get()
    self.input_txt.delete(0, 'end')
    if len(message) > int(self.total_width / 4):
      self.input_txt.configure(width=int(self.total_width * 5 / 10 - len(
        message)))
    else:
      self.input_txt.configure(width=int(self.total_width * 5 / 10))
    self.submit_label.configure(width=len(message))
    self.submit_label.configure(text=message)
    self.queue.put(message)


class MinitensFrame(tk.Frame):
  def __init__(self, parent, *args, **kwargs):
    """
    Special frame used in case of a minitens machine.
    """
    tk.Frame.__init__(self, parent)
    self.grid()
    self.mode = tk.IntVar()
    self.modes = [('stop', 0),
                  ('traction', 1),
                  ('compression', 2),
                  ('cycle', 3)]
    self.create_widgets(**kwargs)
    self.queue = kwargs.get("queue")

  def create_widgets(self, **kwargs):
    self.frame_one = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
    self.minitens_frame_radiobuttons = tk.Frame(self.frame_one)

    # Sequence to compose
    for index, value in enumerate(self.modes):
      tk.Radiobutton(self.minitens_frame_radiobuttons, text=value[0],
                     value=value[1], variable=self.mode).grid(row=index,
                                                              sticky=tk.W)

    self.vitesse_frame = tk.Frame(self.frame_one)
    self.vitesse_parameter = tk.Entry(self.vitesse_frame)
    self.vitesse_parameter.grid(row=1)
    tk.Label(self.vitesse_frame, text="Vitesse(0..255)").grid(row=0)

    self.boucle_frame = tk.Frame(self.frame_one)
    self.boucle_parameter = tk.Entry(self.boucle_frame)
    self.boucle_parameter.grid(row=1)
    tk.Label(self.boucle_frame, text="Temps(ms)").grid(row=0)

    self.buttons_frame = tk.Frame(self.frame_one)
    tk.Button(self.buttons_frame,
              text="SUBMIT",
              bg="green",
              relief="raised",
              height=4, width=10,
              command=lambda: self.update_widgets("SUBMIT")
              ).grid(row=0, column=0)

    tk.Button(self.buttons_frame,
              text="STOP",
              bg="red",
              relief="raised",
              height=4, width=10,
              command=lambda: self.update_widgets("STOP")
              ).grid(row=0, column=1)

    self.minitens_frame_radiobuttons.grid(row=0, column=0)
    self.vitesse_frame.grid(row=0, column=1)
    self.boucle_frame.grid(row=0, column=2)
    self.buttons_frame.grid(row=0, column=4)
    self.frame_one.grid()

    # Limits and pre-loading.
    self.unload_mode = tk.StringVar()
    self.minitens_frame_preload_radiobuttons = tk.Frame(self)
    tk.Radiobutton(self.minitens_frame_preload_radiobuttons,
                   text="HOLD",
                   value="HOLD",
                   variable=self.unload_mode).grid(row=0, sticky=tk.W)
    tk.Radiobutton(self.minitens_frame_preload_radiobuttons,
                   text="UNLOAD",
                   value="UNLOAD",
                   variable=self.unload_mode).grid(row=1, sticky=tk.W)

  def update_widgets(self, *args):
    if args[0] == "STOP":
      message = str({"mode": 0,
                     "vitesse": 255,
                     "boucle": 0})
    else:
      message = str({"mode": self.mode.get(),
                     "vitesse": self.vitesse_parameter.get(),
                     "boucle": self.boucle_parameter.get()})

    self.queue.put(message)


class ArduinoHandler(object):
  def __init__(self, *args, **kwargs):
    """Special class called in a new process, that handles
    connection between crappy and the GUI."""

    def collect_serial(arduino, queue):
      """Collect serial information, in a parallel way."""
      while True:
        queue.put(arduino.readline())

    self.port = args[0]
    self.baudrate = args[1]
    self.queue_process = args[2]
    self.width = args[3]
    self.fontsize = args[4]
    self.frames = args[5]

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
    self.main_loop()

  def init_main_window(self):
    """
    Method to create and place widgets inside the main window.
    """
    self.root = tk.Tk()
    self.root.resizable(width=False, height=False)
    self.root.title("Arduino Minitens")
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
      self.minitens_frame.grid()

  def main_loop(self):
    """
    Main method to update the GUI, collect and transmit information.
    """
    while True:
      try:
        message = self.collect_serial_queue.get(block=True, timeout=0.01)
      except Empty:
        # In case there is a queue timeout
        self.root.update()

      try:
        self.monitor_frame.update_widgets(message)
        # self.minitens_frame.control_force(message)
        self.queue_process.put(message)  # Message is sent to the crappy
        message = ""
        # process.
      except (AttributeError, UnboundLocalError):
        pass

      try:
        to_send = self.submit_serial_queue.get(block=False)
        self.arduino_ser.write(to_send)
      except Empty:
        pass
      self.root.update()


class Arduino(object):
  def __init__(self, *args, **kwargs):
    """
    Main class used ton interface Arduino, its GUI and crappy. For
    reusability, make sure the program inside the arduino sends to the serial
    port a python dictionary formated string.
    Args:
      port: serial port of the arduino.
      baudrate: baudrate defined inside the arduino program.
      width: width of the GUI.
      fontsize: size of the font inside the GUI.
    """
    self.port = kwargs.get("port", "/dev/ttyACM0")
    self.baudrate = kwargs.get("baudrate", 9600)
    self.labels = kwargs.get("labels", None)
    self.frames = kwargs.get("frames", ["monitor", "submit"])

    self.queue_get_data = Queue()
    self.arduino_handler = Process(target=ArduinoHandler,
                                   args=(self.port,
                                         self.baudrate,
                                         self.queue_get_data,
                                         kwargs.get("width", 100),
                                         kwargs.get("fontsize", 11),
                                         self.frames))
    self.handler_t0 = time()
    self.arduino_handler.start()

  def get_data(self, mock=None):
    while True:
      try:
        got = self.queue_get_data.get()
        if len(got) == 0:
          continue
        else:
          retrieved_from_arduino = literal_eval(got)
      except SyntaxError:
        print '[arduino] Skipped data at %.3f sec (Python time)' % (time() -
                                                                    self.handler_t0)
        continue

      if isinstance(retrieved_from_arduino, dict):
        if self.labels:
          ordered = OrderedDict()
          ordered["time(sec)"] = 0.
          for key in self.labels:
            ordered[key] = retrieved_from_arduino[key]
          return time(), ordered
        else:
          return time(), retrieved_from_arduino

  def close(self):
    self.arduino_handler.terminate()
