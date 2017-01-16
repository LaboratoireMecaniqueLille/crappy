# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjacksensor LabJackSensor
# @{

## @file _labJackSensor.py
# @brief  General class for LabJack devices.
# @author Francois Bari
# @version 0.9
# @date 18/08/2016
from __future__ import print_function
from labjack import ljm
from time import time, sleep
from collections import OrderedDict
from multiprocessing import Process, Queue
from Tkinter import Tk, Label
from inspect import ismethod, getmembers
from os import getpid
from ue9 import UE9


class LabJack_UE9(object):
  def __init__(self, *args, **kwargs):
    def get_channel_number(channels):
      """
      register needs to be called with the channel name as int.
      """
      if isinstance(channels[0], str):
        formated_channel = []
        for channel in channels:
          formated_channel.append(int(channel[-1]))
        return formated_channel
      else:
        return channels

    def format_lists(list_to_format, *args):
      """
      In case the user only specifies one parameter, and wants it applied to all inputs.
      """
      list_to_format = [list_to_format] if not isinstance(list_to_format, list) else list_to_format
      if args[0] is not 0:
        if len(list_to_format) == 1:
          return list_to_format * args[0]
        elif len(list_to_format) == args[0]:
          return list_to_format
        else:
          raise TypeError('Wrong Labjack Parameter definition.')
      else:
        return list_to_format
    self.sensor_args = kwargs.get('sensor', None)
    self.actuator_args = kwargs.get('actuator', None)
    if self.sensor_args:
      self.handle = self.open_handle(self.sensor_args)
    elif self.actuator_args:
      self.handle = self.open_handle(self.actuator_args)
    else:
      raise TypeError('Wrong LabJack Parameter definition.')

    if self.sensor_args:
      self.channels = format_lists(self.sensor_args.get('channels', 0), 0)

      self.nb_channels = len(self.channels)
      self.channels = get_channel_number(self.channels)

      self.gain = format_lists(self.sensor_args.get('gain', 1), self.nb_channels)
      self.offset = format_lists(self.sensor_args.get('offset', 0), self.nb_channels)
      self.resolution = format_lists(self.sensor_args.get('resolution', 12), self.nb_channels)

  def open_handle(self, dictionary):
    return UE9()


  def new(self):
    print('new')
    pass

  def start_stream(self):
    print('start stream')
    pass

  def get_data(self, mock=None):
    results = []
    for index, channel in enumerate(self.channels):
      results.append(
        self.handle.getAIN(channel, Resolution=self.resolution[index]) * self.gain[index] + self.offset[index])
    return time(), results

  def get_stream(self):
    print('get stream')
    pass

  def set_cmd(self, cmd, *args):
    print('set cmd')
    pass

  def close(self):
    self.handle.close()
  def close_streamer(self):
    print('close streamer')
    pass


class LabJack_T7(object):
  """Sensor class for LabJack T7 devices."""

  def __init__(self, **kwargs):
    """
    Convert tension value into digital values, on several channels, using LabJack Devices.

    Args:
        mode:                  str.
                               Available modes at the moment :
                               - Single : Output is (measured_value * gain) + offset, can acquire at 1 kHz max.
                               - Thermocouple : Output is a temperature in degree celsius.
                               - Streamer : Output is (measured_value * gain) + offset, can acquire at 100 kHz max.

        channels:              int, str or list of int or str, default = 0
                               The desired input channel(s). If int, will be assumed as "AIN".

        gain:                  float or list of float, default = 1
                               Multiplication gain for each channel. If there is multiple channels
                               for a single gain, it will be applied to all.

        offset:                float, default = 0
                               Add this value for each channel. If there is multiple channels
                               for a single offset, it will be applied to all.

        chan_range:            int or float, default = 10. Can be 10, 1, 0.1 or 0.01, depending on the voltage
                               range
                               to measure. Put the absolute maximum of your expected values.

        resolution:            int, resolution index for each channel (T7 : 0 to 8, T7-PRO : 0 to 12)
                               ~11 to 22 bits depending on the device, the chan_range and the resolution index.
                               higher resolution index = higher resolution, but higher latency.

        scan_rate_per_channel: STREAMER MODE ONLY : int, defines how many scans to perform on each channel
                               during streaming.

        scans_per_read:      STREAMER MODE ONLY : int, defines how many samples to collect during one loop.
                               If undefined, will be assumed as a fraction of sample_rate, determined for performance.
                               BE AWARE : scan_rate_per_channel is for 1 channel, sample_rate, as defined further
                               in this code corresponds to how many samples are collected in total by the
                               labjack device.
                               sample_rate = nb_channels * scan_rate_per_channel

        handle:                If using labjack as I/O device at the same time.
                               Unused for the moment (18/08/2016)

        identifier:            str. Used if multiple labjacks are connected. The identifier could be anything
                               that could define the device : serial number, name, wifi version..
        """

    # super(LabJack, self).__init__()
    def vprint(*args):
      """
      Function used in case of verbosity.
      """
      print('[crappy.technical.Labjack] T7 device, PID', getpid(), *args)

    def open_handle(dictionary):
      """
      Function used only to open handle. For better exception behavior handling.
      """
      try:
        handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, dictionary.get('identifier', 'ANY'))
        return handle
      except ljm.LJMError as e:
        self.vprint('open_handle exception:', e)
        raise
      except:
        raise

    def var_tester(var, nb_channels):
      """Used to check if the user entered correct parameters."""
      var = [var] * nb_channels if isinstance(var, (int, float)) else var
      assert isinstance(var, list) and len(var) == nb_channels, \
        str(var) + "Parameters definition Error: list is not the same length as nb_channels."
      assert False not in [isinstance(var[i], (int, float)) for i in range(nb_channels)], \
        str(var) + "Error: parameters should be int or float."
      return var

    self.verbose = kwargs.get('verbose', False)
    self.vprint = vprint if self.verbose else lambda *args: None

    self.sensor_args = kwargs.get('sensor', None)
    self.actuator_args = kwargs.get('actuator', None)
    self.handle = None

    # while True:
    #   try:
    if self.sensor_args:  # and not self.handle:
      self.handle = open_handle(self.sensor_args)
    elif self.actuator_args:  # and not self.handle:
      self.handle = open_handle(self.actuator_args)
    # elif self.handle:
    #   break
    else:
      self.vprint('Could not open handle.')
      # break
      # except ljm.LJMError.errorCode as error_code:
      #   if error_code == 1239:
      #     self.vprint('Reconnecting...')
      #     pass
      #   else:
      #     raise
    if self.sensor_args:

      self.channels = self.sensor_args.get('channels', 'AIN0')
      self.channels = [self.channels] if not isinstance(self.channels, list) else self.channels
      self.channels = ["AIN" + str(chan) if type(chan) is not str else chan for chan in self.channels]
      self.nb_channels = len(self.channels)

      # To use extended features of ljm libraries:
      self.channels_index_read = [self.channels[chan] + "_EF_READ_A" for chan in range(self.nb_channels)]

      self.chan_range = var_tester(self.sensor_args.get('chan_range', 10), self.nb_channels)
      self.resolution = var_tester(self.sensor_args.get('resolution', 0), self.nb_channels)

      self.gain = var_tester(self.sensor_args.get('gain', 1), self.nb_channels)
      self.offset = var_tester(self.sensor_args.get('offset', 0), self.nb_channels)

      self.mode = self.sensor_args.get('mode', 'single').lower()

      if self.mode == "streamer":
        # Additional variables used in streamer mode only.
        self.a_scan_list = ljm.namesToAddresses(self.nb_channels, self.channels)[0]
        self.scan_rate_per_channel = self.sensor_args.get('scan_rate_per_channel', 1000)

        if self.scan_rate_per_channel * self.nb_channels >= 100000:
          self.scan_rate_per_channel = int(100000 / self.nb_channels)
        self.scans_per_read = self.sensor_args.get('scans_per_read', int(self.scan_rate_per_channel / 10.))
        if self.verbose:
          global queue  # Used to run a dialog box in parallel
          queue = Queue()
          # while True:
          #   try:
          #   break
          # except ljm.LJMError as e:
          #   if e.errorCode == 2605 or e.errorCode == 1239:
          #     pass
          #   else:
          #     raise
          # except Exception:
          #   raise
    self.new()

    if self.actuator_args:
      self.channel_command = self.actuator_args.get('channel', "DAC0")
      self.gain_command = self.actuator_args.get('gain', 1)
      self.offset_command = self.actuator_args.get('offset', 0)

  class DialogBox:
    """
    Dialog box that pops when using streamer function with verbosity.
    """

    def __init__(self, scan_rate_per_channel, scans_per_read):
      self.root = Tk()
      self.root.title('LabJack Streamer')
      self.root.resizable(width=False, height=False)
      self.c2 = []  # List to update
      self.first_column = ['Scan Rate', 'Samples Collecting Rate', 'Chronometer', 'Device Buffer', 'Software Buffer']
      self.second_column = ['%.1f kHz' % (scan_rate_per_channel / 1000.),
                            '%.1f kSamples per read' % (scans_per_read / 1000.), 0.0, 0, 0]
      for row_index, first_column in enumerate(self.first_column):
        Label(self.root, text=first_column, borderwidth=10).grid(row=row_index, column=0)
        self.c2.append(Label(self.root, text=self.second_column[row_index], borderwidth=10))
        self.c2[-1].grid(row=row_index, column=1)
      self.update()

    def update(self):
      """Method to update data inside the dialog box. The window is updated every time data in queue occurs."""
      array = queue.get()
      t0 = array[0]
      while True:
        array[0] = '%.1f' % (array[0] - t0)
        for row_index, value in enumerate(array):
          self.c2[row_index + 2].configure(text=value, borderwidth=10)
        self.root.update()
        array = queue.get()

  def new(self):
    """
    Initialize the device.
    """
    # res_max = 12 if ljm.eReadName(self.handle, "WIFI_VERSION") > 0 else 8  # Test if LabJack is pro or not
    # assert False not in [0 <= self.resolution[chan] <= res_max for chan in range(self.nb_channels)], \
    #   "Wrong definition of resolution index. INDEX_MAX for T7: 8, for T7PRO: 12"
    if self.mode == "single":
      to_write = OrderedDict([
        ("_RANGE", self.chan_range),
        ("_RESOLUTION_INDEX", self.resolution),
        ("_EF_INDEX", 1),  # for applying a slope and offset
        ("_EF_CONFIG_D", self.gain),  # index to set the gain
        ("_EF_CONFIG_E", self.offset)  # index to set the offset
      ])

    elif self.mode == "thermocouple":
      to_write = OrderedDict([
        ("_EF_INDEX", 22),  # for thermocouple measures
        ("_EF_CONFIG_A", 1),  # for degrees C
        ("_EF_CONFIG_B", 60052),  # for type K
        ("_RESOLUTION_INDEX", self.resolution)
      ])

    elif self.mode == "streamer":
      a_names = ["AIN_ALL_RANGE", "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
      a_values = [int(self.chan_range[0]), 0, int(self.resolution[0])]
    else:
      message = 'Error in new: unrecognized mode. Check documentation.'
      self.vprint(message)
      raise TypeError(message)

    if self.mode == "single" or self.mode == "thermocouple":
      a_values = []
      a_names = []
      for chan_iteration in range(self.nb_channels):
        for count, key in enumerate(to_write):
          a_names.append(self.channels[chan_iteration] + to_write.keys()[count])
          if isinstance(to_write.get(key), list):
            a_values.append(to_write.get(key)[chan_iteration])
          else:
            a_values.append(to_write.get(key))

    try:
      ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)
    except ljm.LJMError as e:
      self.vprint('Exception in new creation:', e)
      if e.errorCode == 2605:
        self.close_streamer()
      self.close()
      raise

  def start_stream(self):
    """
    Method to initialize a streaming data.
    """
    try:
      ljm.eStreamStart(self.handle, self.scans_per_read, self.nb_channels,
                       self.a_scan_list, self.scan_rate_per_channel)
      if self.verbose:
        Process(target=self.DialogBox, args=(self.scan_rate_per_channel, self.scans_per_read)).start()

    except ljm.LJMError as e:
      self.vprint('Error in start_stream:', e)
      self.close_streamer()
      raise

  def get_data(self, mock=None):
    """
    Read the signal on all pre-defined channels, one by one.
    """
    try:
      results = ljm.eReadNames(self.handle, self.nb_channels, self.channels_index_read)
      return time(), results

    except ljm.LJMError as e:
      self.vprint('Error in get_data:', e)
      self.close()
      raise

  def get_stream(self):
    """
    Read the device buffer if scan_mode is set.
    """
    try:
      retrieved_from_buffer = ljm.eStreamRead(self.handle)
      results = retrieved_from_buffer[0]
      timer = time()
      if self.verbose:
        queue.put([timer, retrieved_from_buffer[1], retrieved_from_buffer[2]])
      return timer, results

    except Exception:
      self.close_streamer()
      raise

  def set_cmd(self, cmd, *args):
    """
    Convert the tension value to a digital value and send it to the output.
    """
    out = (cmd * self.gain_command) + self.offset_command
    ljm.eWriteName(self.handle, self.channel_command if not args else args[0], out)

  def close(self):
    """
    Close the device.
    """
    try:
      ljm.close(self.handle)
    except ljm.LJMError as e:
      if e.errorCode == 1224:
        pass
      else:
        raise
    except:
      raise
    self.vprint("LabJack device closed")

  def close_streamer(self):
    """
    Special method called if streamer is open.
    """
    while not queue.empty():
      # Flushing the queue
      queue.get_nowait()
    ljm.eStreamStop(self.handle)
    self.close()


class LabJack(object):
  """
  Parent class that loads the one of the above class depending on the device connected.
  """

  def __init__(self, *args, **kwargs):
    """
    The parameters are the same as defined for each device. This class simply inherits from it.
    """
    # def identify_t7(id):
    #   try:
    #     ljm.openS("ANY", "ANY", id)
    #   except ljm.LJMError.errorCode as e:
    #     if e == '1249':
    #       pass
    #     else:
    #       raise

    self.type = kwargs.get('device', None).lower()

    # if not self.type:
    #   identify_t7(kwargs.get('identifier', 'ANY'))
    #

    if self.type == 't7':
      self.sublabjack = LabJack_T7(**kwargs)

    elif self.type == 'ue9':
      self.sublabjack = LabJack_UE9(**kwargs)
    else:
      print('Error: LabJack not recognized.')

    methods = getmembers(self.sublabjack, predicate=ismethod)
    variables = vars(self.sublabjack)

    for key, value in methods:
      setattr(self, key, value)
    for key, value in variables.iteritems():
      setattr(self, key, value)
