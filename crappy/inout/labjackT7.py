# coding: utf-8

from __future__ import print_function,absolute_import,division
from labjack import ljm
from time import time
from threading import Thread
from Queue import Queue
from Tkinter import Tk, Label

from .inout import InOut

class DialogBox:
  """
  Dialog box that pops when using streamer function with verbosity.
  """

  def __init__(self, scan_rate_per_channel, scans_per_read, queue):
    self.root = Tk()
    self.root.title('LabJack Streamer')
    self.root.resizable(width=False, height=False)
    self.c2 = []  # List to update
    self.first_column = ['Scan Rate', 'Samples Collecting Rate',
                         'Chronometer', 'Device Buffer', 'Software Buffer']
    self.second_column = ['%.1f kHz' % (scan_rate_per_channel / 1000.),
                          '%.1f kSamples per read' % (scans_per_read / 1000.),
                          0.0, 0, 0]
    for row_index, first_column in enumerate(self.first_column):
      Label(self.root, text=first_column, borderwidth=10).grid(row=row_index,
                                                               column=0)
      self.c2.append(
        Label(self.root, text=self.second_column[row_index], borderwidth=10))
      self.c2[-1].grid(row=row_index, column=1)
    self.queue = queue
    self.update()

  def update(self):
    """Method to update data inside the dialog box. The window is updated
    every time data in queue occurs."""
    array = self.queue.get()

    t0 = array[0]
    while True:
      array[0] = '%.1f' % (array[0] - t0)
      for row_index, value in enumerate(array):
        self.c2[row_index + 2].configure(text=value, borderwidth=10)
      self.root.update()
      array = self.queue.get()
      if array =='stop':
        break

def open_handle(identifier='ANY'):
  """
  Function used only to open handle. For better exception behavior handling.
  """
  handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY,identifier)
  return handle

def var_tester(var, nb_channels):
  """Used to check if the user entered correct parameters."""
  var = [var] * nb_channels if isinstance(var, (int, float)) else var
  assert isinstance(var, list) and len(var) == nb_channels, \
    str(var) + \
    "Parameter definition Error: list is" \
    " not the same length as nb_channels."
  assert False not in [isinstance(var[i], (int, float)) for i in
                       xrange(nb_channels)], \
    str(var) + "Error: parameter should be int or float."
  return var

class Labjack_T7(InOut):
  """Class for LabJack T7 devices. Used to acquire and set
  analogical datas."""
  def __init__(self, **kwargs):
    """
    Args:
      mode: str. Available modes at the moment :
        single : Output is (measured_value * gain) + offset (~2kHz max.)
        thermocouple : Output is a temperature in degree celsius.
        streamer : Output is (measured_value * gain) + offset (100 kSamples
        max.)

      channels: int, str or list of int or str, default = 0 (AIN0)

      gain: float or list of float, default = 1

      offset: float, default = 0

      chan_range: int or float, default = 10. Can be 10, 1, 0.1  or 0.01,
      depending on the voltage range to measure. Put the absolute maximum of
      your expected values. The higher the range, the fastest the acquisition
      rate is. resolution: int, resolution index for each channel.
      T7 : 1 to 8, T7-PRO : 1 to 12. If 0 is specified, will be 8 (20 bits)
      Check https://labjack.com/support/datasheets/t7/appendix-a-3-1 for more
      information.

      scan_rate_per_channel: STREAMER MODE ONLY : int, defines how many
      scans to perform on each channel during streaming.

      scans_per_read: STREAMER MODE ONLY : int, defines how many

      samples to collect during one loop. If undefined,
      will set to a tenth of sample_rate

      identifier: str. Used if multiple labjacks are connected.
      The identifier could be anything that could define the
      device : serial number, name, wifi version..
        """
    InOut.__init__(self)
    # For now, kwargs like in_gain are eqivalent to gain
    # (it is for consitency with out_gain, out_channels, etc...)
    for arg in kwargs:
      if arg in kwargs and arg.startswith('in_'):
        kwargs[arg[3:]] = kwargs[arg]
        del kwargs[arg]
    for arg,default in [('verbose',False),
                        ('channels','AIN0'),
                        ('gain',1),
                        ('offset',0),
                        ('make_zero',True),
                        ('out_channels',[]),
                        ('out_gain',1),
                        ('out_offset',0),
                        ('mode','single'),
                        ('chan_range', 10),
                        ('resolution',1),
                        ('identifier','ANY'),
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)

    if self.mode == 'streamer':
      for arg,default in [('scan_rate_per_channel',1000),
                          ('scans_per_read',0),
                          ]:
        if arg in kwargs:
          setattr(self,arg,kwargs[arg])
          del kwargs[arg]
        else:
          setattr(self,arg,default)
    assert len(kwargs) == 0,"Labjack_T7 got unsupported arg(s)"+str(kwargs)

    self.vprint = lambda *args:\
        print('[crappy.InOut.LabjackT7]', *args)\
        if self.verbose else lambda *args: None

    self.check_vars()
    self.handle = None

  def check_vars(self):
    """
    Turns the settings into lists of the same length, each index standing for
    one channel.
    if a list is given, simply check the length
    else make a list of the correct length containing only the given value
    """
    self.mode = self.mode.lower()
    self.channels = [self.channels] if not isinstance(self.channels,
                    list) else self.channels
    self.channels = ["AIN" + str(chan) if type(chan) is not str else chan for
                    chan in self.channels]
    self.nb_channels = len(self.channels)
    self.chan_range = var_tester(self.chan_range,self.nb_channels)
    self.resolution = var_tester(self.resolution,self.nb_channels)
    self.make_zero = self.make_zero if isinstance(self.make_zero,list)\
        else [self.make_zero]*self.nb_channels
    assert len(self.make_zero) == self.nb_channels,"Invalid make_zero length"
    self.gain = var_tester(self.gain,self.nb_channels)
    self.offset = var_tester(self.offset,self.nb_channels)
    self.channels_index_read = [self.channels[chan]
                  + "_EF_READ_A" for chan in range(self.nb_channels)]
    if not isinstance(self.out_channels,list):
      self.out_channels = [self.out_channels]
    for i in range(len(self.out_channels)):
      if isinstance(self.out_channels[i],int):
        self.out_channels[i] = 'DAC'+str(self.out_channels[i])
    if not isinstance(self.out_gain,list):
      self.out_gain = [self.out_gain]*len(self.out_channels)
    if not isinstance(self.out_offset,list):
      self.out_offset = [self.out_offset]*len(self.out_channels)
    if self.mode == 'streamer':
      if self.scan_rate_per_channel * self.nb_channels >= 100000:
        self.scan_rate_per_channel = int(100000 / self.nb_channels)
        print("Labjack warning: scan rate too high! Lowering to ",
            self.scan_rate_per_channel)
      if self.scans_per_read == 0:
        self.scans_per_read = int(self.scan_rate_per_channel / 10)

  def open_single(self):
    to_write = [
      ("_RANGE", self.chan_range),
      ("_RESOLUTION_INDEX", self.resolution),
      ("_EF_INDEX", 1),  # for applying a slope and offset
      ("_EF_CONFIG_D", self.gain),  # index to set the gain
      ("_EF_CONFIG_E", self.offset),  # index to set the offset
      ("_SETTLING_US", [0] * self.nb_channels)
    ]
    a_names = []
    a_values = []
    for i,chan in enumerate(self.channels):
      names,values = zip(*to_write)
      names = [chan+n for n in names]
      values = [v[i] if isinstance(v,list) else v for v in values]
      a_names.extend(names)
      a_values.extend(values)
    ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)
    if any(self.make_zero):
      off = self.eval_offset()
      for i,make_zero in enumerate(self.make_zero):
        if make_zero:
          self.offset[i] += off[i]

  def open_streamer(self):
    self.a_scan_list = \
      ljm.namesToAddresses(self.nb_channels, self.channels)[0]
    if self.verbose:
      self.queue = Queue()
    a_names = ["AIN_ALL_RANGE", "STREAM_SETTLING_US",
               "STREAM_RESOLUTION_INDEX"]
    a_values = [int(self.chan_range[0]), 0, int(self.resolution[0])]
    ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)
    self.stream_started = False

  def open_thermocouple(self):
    to_write = [
      ("_EF_INDEX", 22),  # for thermocouple measures
      ("_EF_CONFIG_A", 1),  # for degrees C
      ("_EF_CONFIG_B", 60052),  # for type K
      ("_RESOLUTION_INDEX", self.resolution)
    ]
    a_names = []
    a_values = []
    for i,chan in enumerate(self.channels):
      names,values = zip(*to_write)
      names = [chan+n for n in names]
      values = [v[i] if isinstance(v,list) else v for v in values]
      a_names.extend(names)
      a_values.extend(values)
    ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)

  def open(self):
    self.handle = open_handle(self.identifier)
    # To use extended features of ljm libraries:
    if self.mode == "single":
      self.open_single()
    elif self.mode == "streamer":
      self.open_streamer()
    elif self.mode == "thermocouple":
      self.open_thermocouple()
    else:
      raise IOError("Unknown Labjack mode: "+self.mode)

  def start_stream(self):
    """
    Method to initialize a streaming data.
    """
    try:
      ljm.eStreamStart(self.handle, self.scans_per_read, self.nb_channels,
                       self.a_scan_list, self.scan_rate_per_channel)
    except ljm.LJMError as e:
      print('Error in start_stream:', e)
      self.close_streamer()
      raise
    if self.verbose:
      thread = Thread(target=self.DialogBox, args=(
        self.scan_rate_per_channel, self.scans_per_read, self.queue))
      thread.start()
    self.stream_started = True

  def get_data(self):
    """
    Read the signal on all pre-defined channels, one by one.
    """
    try:
      l = [time()]
      l.extend(ljm.eReadNames(self.handle, self.nb_channels,
                               self.channels_index_read))
      return l
    except ljm.LJMError as e:
      self.vprint('Error in get_data:', e)
      self.close()
      raise

  def get_stream(self):
    """
    Read the device buffer if scan_mode is set.
    """
    if not self.stream_started:
      self.start_stream()
    retrieved_from_buffer = ljm.eStreamRead(self.handle)
    results = retrieved_from_buffer[0]
    timer = time()
    if self.verbose:
      self.queue.put(
        [timer, retrieved_from_buffer[1], retrieved_from_buffer[2]])
    return timer, results

  def set_cmd(self, *cmd):
    """
    Convert the tension value to a digital value and send it to the output.
    """
    for command,channel,gain,offset in zip(
        cmd,self.out_channels,self.out_gain,self.out_offset):
      ljm.eWriteName(self.handle, channel, command*gain+offset)

  def close(self):
    """
    Close the device.
    """
    if self.mode == "streamer":
      self.close_streamer()
    try:
      ljm.close(self.handle)
    except ljm.LJMError as e:
      if e.errorCode != 1224:
        raise
    self.vprint("LabJack device closed")

  def close_streamer(self):
    """
    Special method called if streamer is open.
    """
    if self.verbose:
      while not self.queue.empty():
        self.queue.get_nowait()
      self.queue.put("stop")
    ljm.eStreamStop(self.handle)
