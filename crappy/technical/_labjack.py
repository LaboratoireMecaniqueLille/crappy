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

from labjack import ljm
from time import time, sleep
from sys import exc_info
from os import getpid
from ._meta import io
from collections import OrderedDict
from multiprocessing import Process, Queue
from Tkinter import Tk, Label
from warnings import warn


class LabJack(io.Control_Command):
  """Sensor class for LabJack devices."""

  def __init__(self, sensor={}, actuator={}, **kwargs):
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

    if sensor:
      self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, sensor.get('identifier', 'ANY'))
    elif actuator:
      self.handle = ljm.open(ljm.constants.dtANY, ljm.constants.ctANY, actuator.get('identifier', 'ANY'))
    else:
      print 'NONE'
    if sensor:
      def var_tester(var, nb_channels):
        """Used to check if the user entered correct parameters."""
        var = [var] * nb_channels if isinstance(var, (int, float)) else var
        assert isinstance(var, list) and len(var) == nb_channels, \
          str(var) + "Parameters definition Error: list is not the same length as nb_channels."
        assert False not in [isinstance(var[i], (int, float)) for i in range(nb_channels)], \
          str(var) + "Error: parameters should be int or float."
        return var

      self.channels = sensor.get('channels', 'AIN0')
      self.channels = [self.channels] if not isinstance(self.channels, list) else self.channels
      self.channels = ["AIN" + str(chan) if type(chan) is not str else chan for chan in self.channels]

      self.nb_channels = len(self.channels)
      self.channels_index_read = [self.channels[chan] + "_EF_READ_A" for chan in range(self.nb_channels)]

      self.chan_range = var_tester(sensor.get('chan_range', 10), self.nb_channels)
      self.resolution = var_tester(sensor.get('resolution', 0), self.nb_channels)

      self.gain = var_tester(sensor.get('gain', 1), self.nb_channels)
      self.offset = var_tester(sensor.get('offset', 0), self.nb_channels)

      self.mode = sensor.get('mode', 'single').lower()

      if self.mode == "streamer":
        # Additional variables used in streamer mode only.
        self.a_scan_list = ljm.namesToAddresses(self.nb_channels, self.channels)[0]
        self.scan_rate_per_channel = sensor.get('scan_rate_per_channel', 1000)
        if self.scan_rate_per_channel * self.nb_channels >= 100000:
          self.scan_rate_per_channel = int(100000 / self.nb_channels)
        self.scans_per_read = sensor.get('scans_per_read', int(self.scan_rate_per_channel / 10.))
        global queue  # Used to run a dialog box in parallel
        queue = Queue()
      self.new()
    if actuator:
      self.channel_command = actuator.get('channel', "DAC0")
      self.gain_command = actuator.get('gain', 1)
      self.offset_command = actuator.get('offset', 0)

  class DialogBox:
    """
    Dialog box that pops when using streamer function.
    """

    def __init__(self, scan_rate_per_channel, scans_per_read):
      self.root = Tk()
      self.root.title('LabJack Streamer Information')
      self.root.resizable(width=False, height=False)

      self.first_column = \
        ['Scan Rate', 'Samples Collecting Rate', 'Chronometer', 'Device Buffer', 'Software Buffer']
      self.second_column = \
        ['%.1f kHz' % (scan_rate_per_channel / 1000.), '%.1f kSamples per read' % (scans_per_read / 1000.), 0.0, 0, 0]
      for row_index, first_column in enumerate(self.first_column):
        Label(self.root, text=first_column, borderwidth=10).grid(row=row_index, column=0)
        Label(self.root, text=self.second_column[row_index], borderwidth=10).grid(row=row_index, column=1)
      self.update()

    def update(self):
      """Method to update data inside the dialog box. The window is updated every time data in queue occurs."""
      print "LabJack Sensor / Streamer Dialog Box PID:", getpid()
      try:
        array = queue.get()
        t0 = array[0]
        while True:
          array[0] = '%.1f' % (array[0] - t0)
          for row_index, value in enumerate(array):
            Label(self.root, text=value, borderwidth=10).grid(row=row_index + 2, column=1)
          self.root.update()
          array = queue.get()
      except KeyboardInterrupt:
        pass
      except Exception:
        raise

  def new(self):
    """
    Initialize the device.single
    """
    try:
      res_max = 12 if ljm.eReadName(self.handle, "WIFI_VERSION") > 0 else 8  # Test if LabJack is pro or not
      assert False not in [0 <= self.resolution[chan] <= res_max for chan in range(self.nb_channels)], \
        "Wrong definition of resolution index. INDEX_MAX for T7: 8, for T7PRO: 12"

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
        raise Exception("Unrecognized mode. Check documentation.")

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
      ljm.eWriteNames(self.handle, len(a_names), a_names, a_values)
    except ljm.LJMError:
      self.close()
      raise exc_info()[1]
    except:
      self.close()
      raise

  def start_stream(self):
    """
    Method to initialize a streaming data.
    """
    try:
      ljm.eStreamStart(self.handle, self.scans_per_read, self.nb_channels,
                       self.a_scan_list, self.scan_rate_per_channel)
      Process(target=self.DialogBox, args=(self.scan_rate_per_channel, self.scans_per_read)).start()
    except ljm.LJMError:
      self.close()
      raise exc_info()[1]
    except Exception:
      self.close()
      raise

  def get_data(self, mock=None):
    """
    Read the signal on all pre-defined channels, one by one.
    """
    try:
      results = ljm.eReadNames(self.handle, self.nb_channels, self.channels_index_read)
      return time(), results

    except KeyboardInterrupt:
      self.close()
      pass

    except Exception:
      print(exc_info()[1])
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
      queue.put([timer, retrieved_from_buffer[1], retrieved_from_buffer[2]])
      return timer, results

    except KeyboardInterrupt:
      self.close()
      pass
    except ljm.LJMError:
      ljm.eStreamStop(self.handle)
      self.close()
      raise exc_info()[1]
    except Exception:
      self.close()
      raise exc_info()[1]

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
      self.close_streamer()
    except:
      pass
    finally:
      ljm.close(self.handle)
    print "LabJack device closed"

  def close_streamer(self):
    """
    Special method called if streamer is open.
    """
    # Flushing the queue
    while not queue.empty():
      queue.get_nowait()
    ljm.eStreamStop(self.handle)
    # In order to assure the streamer is turned off.
    while ljm.eReadName(self.handle, "STREAM_ENABLE"):
      sleep(0.5)
      ljm.eStreamStop(self.handle)
