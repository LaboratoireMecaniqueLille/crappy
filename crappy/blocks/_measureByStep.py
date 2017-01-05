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
from _masterblock import MasterBlock
import time
from collections import OrderedDict
from ..links._link import TimeoutError
import threading
from Queue import Queue
import sys


class MeasureByStep(MasterBlock):
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
    super(MeasureByStep, self).__init__()
    self.sensor = sensor
    assert sensor, 'ERROR in MeasureByStep: no sensor defined.'
    self.labels = kwargs.get('labels', ["time(sec)"] + [self.sensor.channels])
    self.freq = kwargs.get('freq', None)
    self.verbose = kwargs.get('verbose', None)
    if self.verbose:
      self.nb_acquisitions = 0.
      global queue, last_len
      last_len = None
      queue = Queue()

  def print_time(self):
    def reprint(*args):
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
      reprint('Freq:', nb_points1 - nb_points0, 'Hz')
      nb_points0 = nb_points1

  def temporization(self, timer):
    while time.time() - timer < 1. / self.freq:
      time.sleep(1. / (100 * self.freq))
    pass

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
        elapsed = 5.
      while True:
        if trigger == "internal":
          if self.freq:  # timing loop
            self.temporization(time.time())
          sensor_epoch, sensor_values = self.sensor.get_data("all")
          chronometer = sensor_epoch - self.t0
          sensor_values.insert(0, chronometer)
        if trigger == "external":
          if self.inputs[0].recv():  # wait for a signal
            pass
          sensor_epoch, sensor_values = self.sensor.get_data("all")
          chronometer = sensor_epoch - self.t0
          sensor_values.insert(0, chronometer)

        results = OrderedDict(zip(self.labels, sensor_values))
        try:
          for output in self.outputs:
            output.send(results)
          if self.verbose:
            self.nb_acquisitions += 1

            if chronometer - elapsed > 1.:
              queue.put(self.nb_acquisitions)
              elapsed = chronometer
        except TimeoutError:
          raise
        except AttributeError:  # if no outputs
          pass

    except (Exception, KeyboardInterrupt) as e:
      print("Exception in measureByStep :", e)
      self.sensor.close()
