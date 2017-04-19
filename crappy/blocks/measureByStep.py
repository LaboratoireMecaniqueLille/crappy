# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MeasureByStep MeasureByStep
# @{

## @file measureByStep.py
# @brief Streams value measured on a card through a Link object.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from __future__ import print_function, division

from time import time,sleep
import threading
from Queue import Queue
import sys

from .masterblock import MasterBlock
from ..inout import in_list

class MeasureByStep(MasterBlock):
  """
  Streams value measured on a card through a Link object.
  """

  def __init__(self, sensor_name, **kwargs):
    """
    This streamer read the value on all channels ONE BY ONE and send
    the values through a Link object.

    It is slower than StreamerComedi, but works on every USB driver.
    It also works on LabJack devices.

    It can be triggered by a Link sending boolean (through "add_input"
    method), or internally by defining the frequency.

    Args:
        sensor:     sensor object
                    See sensor.sensor documentation. Tested for
                    LabJackSensor and ComediSensor.
        labels:     list, optional
                    The labels you want on your output data. If None,
                    will be time(sec) as first arg, and the
                    channel number as additional args.
        freq :      float or int, optional
                    Wanted acquisition frequency. If none, will be
                    at the software looping speed.
    """
    MasterBlock.__init__(self)
    self.sensor_name = sensor_name
    assert sensor_name in in_list,"Unknown sensor: "+sensor_name
    for arg,default in [('freq',None),
                        ('verbose',False),
                        ('labels',['t(s)']),
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    self.sensor_kwargs = kwargs

  def print_time(self):
    data = 1
    while data != 'stop':
      data = self.queue.get()
      sys.stdout.write('\r[MeasureByStep] Samples/Sec: {}'.format(data))
      sys.stdout.flush()

  def prepare(self):
    """
    Block called before main.
    """
    if self.verbose:
      self.prepare_verbosity()
    self.trigger = "internal" if len(self.inputs) == 0 else "external"
    self.sensor = in_list[self.sensor_name](**self.sensor_kwargs)
    self.sensor.open()
    data = self.sensor.get_data()
    while len(self.labels) < len(data):
      self.labels.append(str(len(self.labels)))

  def prepare_verbosity(self):
    self.nb_loops = 0
    self.last_print = time()
    self.queue = Queue()
    printer = threading.Thread(target=self.print_time)
    printer.start()
    self.last_t = time()

  def print_verbosity(self, timer):
    self.nb_acquisitions += 1
    self.time_interval = timer - self.elapsed

    if self.time_interval >= 1.:
      self.elapsed = timer
      self.queue.put(self.nb_acquisitions)
      self.nb_acquisitions = 0

  def loop(self):
    if self.trigger == "external":
      self.inputs[0].recv(blocking=True)
    if self.freq:
      t = time()
      while t < self.last_t + 1/self.freq:
        sleep((self.last_t + 1/self.freq - t)/10)
        t = time()
      self.last_t = t
    data = self.sensor.get_data()
    data[0] -= self.t0
    self.send(data)

    if self.verbose:
      self.nb_loops += 1
      t = time()
      if (t - self.last_print > 1):
        self.queue.put(self.nb_loops/(t - self.last_print))
        self.nb_loops = 0
        self.last_print = t

  def finish(self):
    if hasattr(self,"queue"):
      self.queue.put("stop")
    self.sensor.close()
