import serial
from threading import Thread
from Tkinter import *
from Queue import Queue as Queue_threading, Empty
from time import sleep, time
from collections import OrderedDict
from multiprocessing import Process, Queue


class ArduinoProcess(object):
  """
  Dialog box for arduino monitor : shows the received from serial port,
  and permits to transmit information. This is done in a separate process,
  called by the main class.
  """

  def __init__(self, arduino, queue_recv, program, title_window):
    """
    :param arduino: Arduino serial port opened.
    :param queue_recv: Queue to transmit information from serial to crappy.
    :param program: Program used, to adapt the GUI.
    :param title_window: GUI window title.
    """

    def collect_serial(arduino, queue):
      """
      Serial data are collected in a separated thread, then put in a
      queue to communicate with the main thread.
      """
      while True:
        queue.put(arduino.readline())

    self.arduino = arduino
    self.queue_from_arduino = queue_recv
    self.queue_collect_serial = Queue_threading()
    self.init_main_window(title_window)
    if program == 'minitens':
      self.init_minitens_layout()
    collect_serial_thread = Thread(target=collect_serial,
                                   args=(self.arduino,
                                         self.queue_collect_serial))
    collect_serial_thread.daemon = True
    collect_serial_thread.start()
    self.main()

  def init_main_window(self, title_window):

    # Main window
    self.root = Tk()
    self.root.title('Arduino Monitor on port %s' % title_window[0])
    self.root.resizable(width=False, height=True)

    # Frame to display received from serial
    self.frame_serial_txt = Frame(self.root)
    Label(self.frame_serial_txt, text='Serial port output, baudrate = %s' %
                                      title_window[1]).pack(side=TOP)
    self.serial_txt = Text(self.frame_serial_txt, relief="sunken")
    self.serial_txt.pack(side=BOTTOM)
    self.frame_serial_txt.pack(side=TOP, padx=5, pady=5)

    # Frame for entry to submit to the serial port
    self.frame_input = Frame(self.root)
    self.input_txt = Entry(self.frame_input)
    self.input_txt.pack(side=LEFT, padx=5, pady=5)

    # Submit button (enter key works as well)
    self.submit_button = Button(self.frame_input,
                                text='Submit',
                                command=self.submit_serial)

    self.submit_button.pack(side=RIGHT, padx=5, pady=5)
    self.root.bind('<Return>', self.submit_serial)

    # Label to show the previous entered command
    self.submit_label = Label(self.frame_input, text='')
    self.submit_label.pack(side=LEFT, padx=5, pady=5)
    self.frame_input.pack(side=BOTTOM)
    # Handle on closing event
    self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

  def init_minitens_layout(self):

    self.input_txt.configure(state='disable')
    self.minitens_frame = Frame(self.root)
    self.minitens_frame_radiobuttons = Frame(self.minitens_frame)
    self.modes = OrderedDict([('stop', 0,),
                              ('traction', 1),
                              ('compression', 2),
                              ('cycle', 3)])

    # Radio buttons for the mode
    for index, value in self.modes.iteritems():
      Radiobutton(self.minitens_frame_radiobuttons, text=index,
                  value=value).pack(anchor=W)
    self.minitens_frame_radiobuttons.pack(side=LEFT, padx=10)

    # Set the speed parameter
    Label(self.minitens_frame, text='vitesse').pack(side=LEFT)
    self.vitesse_parameter = Entry(self.minitens_frame, text='vitesse')
    self.vitesse_parameter.pack(side=LEFT, padx=10)

    # Set the time parameter
    Label(self.minitens_frame, text='temps').pack(side=LEFT)
    self.temps_parameter = Entry(self.minitens_frame, text='temps')
    self.temps_parameter.pack(side=LEFT, padx=10)
    self.minitens_frame.pack(side=BOTTOM)

  def main(self):
    """
    Main loop.
    """
    while True:
      try:
        value = self.queue_collect_serial.get(block=False)
        self.update(value)
      except Empty:
        sleep(0.01)
        self.root.update()

  def update(self, value):
    self.queue_from_arduino.put(value)  # To communicate with the crappy
    # process.
    self.serial_txt.insert("0.0", value + '\n')  # To insert at the top
    self.root.update()

  def submit_serial(self, *args):
    """
    Executed if OK key is clicked, or enter key is hit
    """
    message = self.input_txt.get()
    self.input_txt.delete(0, 'end')
    self.submit_label.configure(text=message)
    self.arduino.write(message)
    # self.queue_send.put(message)

  def on_closing(self):
    self.root.destroy()


class Arduino(object):
  def __init__(self, *args, **kwargs):
    # Specific to the GUI and the arduino class, used for serial communication.
    self.title_window = [kwargs.get('port', '/dev/ttyACM0'),
                         kwargs.get('baudrate', 9600)]
    self.arduino_ser = serial.Serial(port=self.title_window[0],
                                     baudrate=self.title_window[1])
    self.monitor = kwargs.get('monitor', False)
    self.program = kwargs.get('program', None)
    self.queue_serial_from_arduino = Queue()  # Acquired with readline()
    # self.queue_serial_to_arduino = Queue()  # Set with write()

    thread_arduino_serial = Process(target=ArduinoProcess,
                                    args=(self.arduino_ser,
                                          self.queue_serial_from_arduino,
                                          self.program,
                                          self.title_window))
    thread_arduino_serial.daemon = True
    thread_arduino_serial.start()

    # Specific to CRAPPY
    self.channels = ['Serial']

  def get_data(self, mock=None):
    try:
      received_from_arduino = self.queue_serial_from_arduino.get()
      return time(), map(int, received_from_arduino.split(' '))
    except ValueError:
      return time(), [0, 0, 0, 0]

  def close(self):
    self.arduino_ser.close()