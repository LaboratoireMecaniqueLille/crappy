# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MeasureByStep MeasureByStep
# @{

## @file _measureByStep.py
# @brief Streams value measured on a card through a Link object.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from __future__ import print_function
from _compacterblock import CompacterBlock
import time
from collections import OrderedDict
from ..links._link import TimeoutError
import threading
from Queue import Queue
import sys


class MeasureByStep(CompacterBlock):
  """
  Streams value measured on a card through a Link object.
  """

  def __init__(self, sensor, *args, **kwargs):
    """
    This streamer read the value on all channels ONE BY ONE and send the values through a Link object.

    It is slower than StreamerComedi, but works on every USB driver.
    It also works on LabJack devices.

    It can be triggered by a Link sending boolean (through "add_input" method),
    or internally by defining the frequency.

    Args:
        sensor:     sensor object
                    See sensor.sensor documentation. Tested for LabJackSensor and ComediSensor.
        labels:     list, optional
                    The labels you want on your output data. If None, will be time(sec) as first arg, and the
                    channel number as additional args.
        freq :      float or int, optional
                    Wanted acquisition frequency. If none, will be at the software looping speed.
    """
    self.sensor = sensor
    assert sensor, 'ERROR in MeasureByStep: no sensor defined.'
    self.labels = kwargs.get('labels', ["time(sec)"] + self.sensor.channels)
    CompacterBlock.__init__(self, labels=self.labels, compacter=kwargs.get("compacter", 1))
    self.freq = float(kwargs.get('freq', None))
    self.verbose = kwargs.get('verbose', False)
    if self.verbose:
      self.nb_acquisitions = 0.
      global queue, last_len, time_interval
      time_interval = 1.
      last_len = None
      queue = Queue()

  def print_time(self):
    def reprint(*args):
      """
      Method to update printed value, instead of print a new one.
      """
      global last_len
      s = " ".join([str(i) for i in args])
      s = s.split("\n")[0]
      l = len(s)
      if last_len is not None:
        s += " " * (last_len - l)
        sys.stdout.write("\033[F")
      last_len = l
      print(s)

    nb_points0 = 0.
    while True:
      nb_points1 = queue.get()
      reprint('Freq:', '%.2f' % ((nb_points1 - nb_points0) / time_interval), 'Hz')
      nb_points0 = nb_points1

  def temporization(self, timer):
    t_a = time.time()
    while time.time() - t_a < timer:
      time.sleep(timer / 100)

  def main(self):
    """
    Main loop for MeasureByStep. Retrieves data at specified frequency (or software looping speed) from specified
    sensor, and sends it to a crappy link.
    """
    try:
      trigger = "internal" if len(self.inputs) == 0 else "external"
      if self.verbose:
        printer = threading.Thread(target=self.print_time)
        printer.daemon = True
        printer.start()
      elapsed = 0.
      while True:
        t_before_acq = time.time()
        if trigger == "internal":
          pass
        elif trigger == "external":
          if self.inputs[0].recv(blocking=True):  # wait for a signal
            pass
        sensor_epoch, sensor_values = self.sensor.get_data("all")
        chronometer = sensor_epoch - self.t0
        sensor_values.insert(0, chronometer)

        try:
          self.send(sensor_values)
          if self.verbose:
            self.nb_acquisitions += 1
          if chronometer - elapsed >= 1.:
            global time_interval
            time_interval = chronometer - elapsed
            elapsed = chronometer
            if self.verbose:
              queue.put(self.nb_acquisitions)
          pass
        except TimeoutError:
          raise
        except AttributeError:  # if no outputs
          pass
        t_after_acq = time.time()
        if self.freq and t_after_acq - t_before_acq < 1 / self.freq:
          self.temporization(1 / self.freq - (t_after_acq - t_before_acq))
        else:
          pass

    except (Exception, KeyboardInterrupt) as e:
      print("Exception in measureByStep :", e)
      self.sensor.close()
      raise
