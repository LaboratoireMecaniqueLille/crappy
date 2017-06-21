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

from .inout import InOut


class MonitorFrame(tk.Frame):
  def __init__(self, parent, **kwargs):
    """
    A frame that displays everything entering the serial port.

    args:
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

  def update_widgets(self, arg):
    if self.enabled_checkbox.get():
      self.serial_monitor.insert("0.0", arg)  # To insert at the top


class SubmitSerialFrame(tk.Frame):
  def __init__(self, parent, **kwargs):
    """
    Frame that permits to submit to the serial port of arduino.
    
    args:
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

  def update_widgets(self):
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
  def __init__(self, parent, **kwargs):
    """
    Special frame used in case of a minitens machine.
    """
    tk.Frame.__init__(self, parent)
    self.grid()
    self.mode = tk.IntVar()
    self.create_widgets(**kwargs)
    self.queue = kwargs.get("queue")

  def add_button(self, widgets_dict, frame, text='Button', bg='white',
                 command=None):
    widgets_dict[text] = tk.Button(frame,
                                   text=text,
                                   bg=bg,
                                   relief="raised",
                                   height=2, width=10,
                                   command=lambda: self.submit_command(command)
                                   )

  def add_label(self, widgets_dict, frame, text='label', relief='raised',
                font=("Courier bold", 12)):
    widgets_dict[text] = (tk.Label(frame, text=text, relief=relief,
                                   font=font))

  def add_entry(self, widgets_dict, frame, entry_name):
    widgets_dict[entry_name] = tk.Entry(frame)

  def add_listbox(self, widgets_dict, frame, name):
    widgets_dict[name] = tk.Listbox(frame)

  def create_widgets(self, **kwargs):
    """
    Frames organization
      Frame_displayer: to display effort and displacement values.
      Frame_position: to start and stop the motor, according to pre-defined
      values.
      Frame_cycle: cycle generator.
    """
    self.frame_displayer = tk.Frame(self,
                                    relief=tk.SUNKEN,
                                    borderwidth=1)

    self.frame_position = tk.Frame(self,
                                   relief=tk.SUNKEN,
                                   borderwidth=1)
    self.frame_limits = tk.Frame(self,
                                 relief=tk.SUNKEN,
                                 borderwidth=1)
    self.frame_cycles = tk.Frame(self,
                                 relief=tk.SUNKEN,
                                 borderwidth=1)

    self.frame_displayer_widgets = OrderedDict()
    self.frame_position_widgets = OrderedDict()
    self.frame_limits_widgets = OrderedDict()
    self.frame_cycles_widgets = OrderedDict()
    self.lim_behavior_widgets = OrderedDict()

    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text="Effort(N)")
    self.add_button(self.frame_displayer_widgets,
                    self.frame_displayer,
                    text='tare_effort',
                    command='tare_effort')
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='effort',
                   font=("Courier bold", 48))
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='Position(mm)')

    self.add_button(self.frame_displayer_widgets,
                    self.frame_displayer,
                    text='tare_position',
                    command='tare_position')
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='position',
                   font=("Courier bold", 48))
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='l0')
    self.add_label(self.frame_displayer_widgets,
                   self.frame_displayer,
                   text='position_prct',
                   font=("Courier bold", 48))
    self.add_entry(self.frame_displayer_widgets,
                   self.frame_displayer,
                   entry_name='l0_entry')

    for command in ["TRACTION", "STOP", "COMPRESSION"]:
      self.add_button(self.frame_position_widgets, self.frame_position,
                      text=command, bg='green', command=command)
    labels_limits = ['Limites',
                     'Effort',
                     'Position',
                     'Position_prct',
                     'Haute',
                     'Basse']
    labels_entries = ['lim_haute_effort',
                      'lim_basse_effort',
                      'lim_haute_position',
                      'lim_basse_position',
                      'lim_haute_position_prct',
                      'lim_basse_position_prct']

    self.lim_behavior = tk.IntVar()
    self.lim_behavior_widgets["maintien"] = tk.Radiobutton(self.frame_limits,
                                                           text='Maintien',
                                                           value=0,
                                                           variable=self.lim_behavior)

    self.lim_behavior_widgets["decharge"] = tk.Radiobutton(self.frame_limits,
                                                           text='Cycles :',
                                                           value=1,
                                                           variable=self.lim_behavior)
    self.add_entry(self.lim_behavior_widgets,
                   self.frame_limits,
                   'nb_cycles')

    self.add_label(self.frame_limits_widgets,
                   self.frame_limits,
                   text='CONSIGNES',
                   font=("Courier bold", 28))

    for text in labels_limits:
      self.add_label(self.frame_limits_widgets,
                     self.frame_limits,
                     text=text)
    for entry in labels_entries:
      self.add_entry(self.frame_limits_widgets,
                     self.frame_limits,
                     entry_name=entry)

    self.add_label(self.frame_cycles_widgets,
                   self.frame_cycles,
                   text="GENERATEUR DE CYCLES",
                   font=("Courier bold", 12))

    labels_cycles = ['#',
                     'Effort min',
                     'Effort max',
                     "Position min",
                     "Position max"]
    for label in labels_cycles:
      self.add_label(self.frame_cycles_widgets,
                     self.frame_cycles,
                     text=label)

    for label in labels_cycles:
      self.add_entry(self.frame_cycles_widgets,
                     self.frame_cycles,
                     label + '_entry')

    self.add_listbox(self.frame_cycles_widgets,
                     self.frame_cycles,
                     name='list_cycles')

    self.frame_displayer_widgets["Effort(N)"].grid(row=0,
                                                   column=0,
                                                   columnspan=3,
                                                   sticky=tk.W)
    self.frame_displayer_widgets['tare_effort'].grid(row=1,
                                                     column=0,
                                                     sticky=tk.W)
    self.frame_displayer_widgets['effort'].grid(row=1,
                                                column=1,
                                                columnspan=2)

    self.frame_displayer_widgets['Position(mm)'].grid(row=2,
                                                      column=0,
                                                      columnspan=3,
                                                      sticky=tk.W)
    self.frame_displayer_widgets['tare_position'].grid(row=3,
                                                       column=0,
                                                       sticky=tk.W)
    self.frame_displayer_widgets['position'].grid(row=3, column=1,
                                                  sticky=tk.W)
    self.frame_displayer_widgets['position_prct'].grid(row=3, column=2,

                                                       sticky=tk.W)
    self.frame_displayer_widgets['l0'].grid(row=4, column=0, sticky=tk.W)
    self.frame_displayer_widgets['l0_entry'].grid(row=4, column=1, sticky=tk.W)
    self.frame_limits_widgets["CONSIGNES"].grid(row=0, columnspan=3)

    # labels_limits = ['Limites',
    #                  'Effort',
    #                  'Position',
    #                  'Position_prct',
    #                  'Haute',
    #                  'Basse']
    labels_entries = ['lim_haute_effort',
                      'lim_haute_position',
                      'lim_haute_position_prct',
                      'lim_basse_effort',
                      'lim_basse_position',
                      'lim_basse_position_prct']

    for id, widget in enumerate(self.frame_position_widgets):
      self.frame_position_widgets[widget].grid(row=0 + 1, column=id)

    for id, widget in enumerate(labels_limits[:4]):
      self.frame_limits_widgets[widget].grid(row=id + 1, column=0)

    for id, widget in enumerate(labels_limits[4:]):
      self.frame_limits_widgets[widget].grid(row=1, column=1 + id)

    for id, widget in enumerate(labels_entries[:3]):
      self.frame_limits_widgets[widget].grid(row=2 + id, column=1)

    for id, widget in enumerate(labels_entries[3:]):
      self.frame_limits_widgets[widget].grid(row=2 + id, column=2)

    for id, widget in enumerate(self.lim_behavior_widgets):
      self.lim_behavior_widgets[widget].grid(row=5, column=0 + id, columnspan=1)

    self.frame_cycles_widgets["GENERATEUR DE CYCLES"].grid(row=0, columnspan=4)
    for id, widget in enumerate(labels_cycles):
      self.frame_cycles_widgets[widget].grid(row=1, column=id)
    for id, widget in enumerate(labels_cycles):
      self.frame_cycles_widgets[widget + '_entry'].grid(row=2, column=id)
    self.frame_cycles_widgets["list_cycles"].grid(row=3, columnspan=4)

    self.frame_displayer.grid(row=0, column=0)
    self.frame_position.grid(row=1, column=0)
    self.frame_limits.grid(row=2, column=0)
    # self.frame_cycles.grid(row=0, column=1)

  def submit_command(self, arg):

    lim_haute_eff = self.frame_limits_widgets['lim_haute_effort'].get()
    lim_basse_eff = self.frame_limits_widgets['lim_basse_effort'].get()
    lim_haute_pos = self.frame_limits_widgets['lim_haute_position'].get()
    lim_basse_pos = self.frame_limits_widgets['lim_basse_position'].get()
    lim_haute_pos_prct = self.frame_limits_widgets[
      'lim_haute_position_prct'].get()
    lim_basse_pos_prct = self.frame_limits_widgets[
      'lim_basse_position_prct'].get()
    nb_cycles = self.lim_behavior_widgets["nb_cycles"].get()

    print('entree et def des limites', lim_basse_pos, lim_haute_eff)

    dico = {"consigne": 1,
            "vitesse": 200,
            "lim_haute_effort": lim_haute_eff,
            "lim_basse_effort": lim_basse_eff,
            "lim_haute_position": lim_haute_pos,
            "lim_basse_position": lim_basse_pos,
            "lim_cycles": nb_cycles,
            "lim_haute_position_prct": lim_haute_pos_prct,
            "lim_basse_position_prct": lim_basse_pos_prct,
            "l0": self.frame_displayer_widgets["l0_entry"].get()
            }

    if arg == "STOP":
      dico = {"sens": 0, "consigne": 1}
    elif arg == "TRACTION":
      dico["sens"] = 1
      print('traction')
    elif arg == "COMPRESSION":
      dico["sens"] = -1
      print('compression')

    if arg == "tare_position":
      dico = {"tare_position": 1}
    elif arg == "tare_effort":
      dico = {'tare_effort': 1}

    message = str(dico)
    print('envoye', message)
    self.queue.put(message)

  def update_widgets(self, message):
    try:
      dico = literal_eval(message)
      self.frame_displayer_widgets["position"].configure(text=dico[
        "position_abs"])
      self.frame_displayer_widgets["effort"].configure(text=dico[
        "effort"])
      self.frame_displayer_widgets["position_prct"].configure(text=dico[
        "position_prct"])
    except (SyntaxError, EOFError, ValueError):
      pass


class ArduinoHandler(object):
  def __init__(self, port, baudrate, queue_process, width, fontsize, frames):
    """Special class called in a new process, that handles
    connection between crappy and the GUI."""

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
        if "monitor" in self.frames:
          self.monitor_frame.update_widgets(message)
        if "minitens" in self.frames:
          self.minitens_frame.update_widgets(message)
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
    self.port = kwargs.get("port", "/dev/ttyACM0")
    self.baudrate = kwargs.get("baudrate", 9600)
    self.labels = kwargs.get("labels", None)
    self.frames = kwargs.get("frames", ["monitor", "submit"])
    self.width = kwargs.get("width", 100)
    self.fontsize = kwargs.get("fontsize", 11)

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
    while True:
      try:
        retrieved_from_arduino = literal_eval(self.queue_get_data.get())
        # print('retrieved from arduino:', retrieved_from_arduino)
        if isinstance(retrieved_from_arduino, dict):
          if self.labels:
            ordered = OrderedDict()
            ordered["time(sec)"] = 0.
            for key in self.labels:
              ordered[key] = retrieved_from_arduino[key]
            return ordered
          else:
            return retrieved_from_arduino

        print('ok')
      except EOFError:
        continue
      except Exception:
        # print '[arduino] Skipped data at %.3f sec (Python time)' % (time() -
        #                                                        self.handler_t0)
        continue

  def close(self):
    self.arduino_handler.terminate()
