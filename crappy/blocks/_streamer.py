# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Streamer Streamer
# @{

## @file _streamer.py
# @brief Send a stream of data.
# @author Corentin Martel
# @version 0.1
# @date 13/07/2016

from _masterblock import MasterBlock
from collections import OrderedDict
from ..links._link import TimeoutError
from os import getpid
import numpy as np
from multiprocessing import Process, Queue
from time import sleep, time


class Streamer(MasterBlock):
  """
  Send a stream of data. Works with LabJack T7 (100 kSamples max), OpenDAQ (1kHz per channel).

  """

  def __init__(self, sensor=None, labels=None, *args, **kwargs):
    """
    Use it for LabJack streaming.
    You can directly link output data to a graph/save without going through a compacter.
    Args:
        sensor: LabJack sensor instance
                Will read the streaming buffer of the LabJack device as defined at
                the instance creation.
        labels : list of str, default = name of used sensor channels output labels.
        mean : int, number to shrink data. For instance, if 10 000 values are in input, mean = 10 will send
        in output 1000 values.

    This block does the time vector reconstruction, then assemble it with the results matrix read from the
    LJM Buffer or OpenDAQ buffer.
    """
    super(Streamer, self).__init__()
    assert sensor, "No input sensor defined."
    self.sensor = sensor
    self.labels = labels if labels else ['time(sec)'] + self.sensor.channels
    self.mean = kwargs.get('mean', None)

    if type(self.sensor).__name__ == 'LabJack':
      """"
      Additionnal values used only for LabJack Streaming: queue for time vector reconstruction.
      """
      global queue
      queue = Queue(2)
    elif type(self.sensor).__name__ == 'OpenDAQ':
      """
      Additionnal values used only for OpenDAQ Streaming: a variable used for time vector reconstruction to keep track.
      """
      self.current_length = 0.

  def time_vector(self):
    """
    Time vector reconstruction, made in another process. Creates the time vector of the
    length = nb_samples, then puts it in a queue.
    """
    sample_number = 0
    while True:
      """
      LJM Buffer sends a list of points with fixed length at a fixed interval, so we're creating the time vector
      consequently.
      This method sends a time_vector of N points equally dispatched on 1 second.
      """
      try:
        ratio = self.sensor.scans_per_read / float(self.sensor.scan_rate_per_channel)
        nb_points = self.sensor.scans_per_read if not self.mean else self.sensor.scans_per_read / self.mean
        time_vector = np.linspace(sample_number * ratio, (sample_number + 1 - 1 / float(nb_points)) * ratio,
                                  nb_points)
        time_vector = np.around(time_vector, 5).tolist()
        queue.put(time_vector)
        sample_number += 1
      except:
        while not queue.empty():  # flush the queue after leaving the loop
          queue.get_nowait()
        break

  def reshape(self, nparray, n):
    """
    Method to average data.
    """
    reshaped = []
    print 'np shape nparray:', np.shape(nparray)
    for length in xrange(np.shape(nparray)[1] / n):
      reshaped.append(np.mean(nparray[:, length * n: int((length + 1 - 1 / float(n)) * n)], axis=1).tolist())
    return np.array(reshaped)

  def get_stream_from_labjack(self):
    """
    Method to get stream from labjack devices.
    The LJM Buffer sends a list of scan_rate_per_channel * nb_channels at every reading, interlaced.
    e.g, if 3 channels are in the scan list, results are : [chan1 chan2 chan3 chan1 chan2 chan3 chan1 ....]
    """
    results = []
    retrieved = self.sensor.get_stream()[1]
    deinterlaced = np.array([retrieved[each::self.sensor.nb_channels] for each in xrange(self.sensor.nb_channels)])
    if self.mean:
      deinterlaced = self.reshape(deinterlaced, self.mean)
    for each in xrange(self.sensor.nb_channels):
      liste_temp = self.sensor.gain[each] * deinterlaced[each] + self.sensor.offset[each]  # Applying gain and offset
      results.append(liste_temp.tolist())
    results.insert(0, queue.get())
    return results

  def get_stream_from_opendaq(self):
    """
    Method to get stream from opendaq devices.
    OpenDAQ collects samples at approximately 1 kHz for every channel, and has to be retrieved with the dq.read()
    method. this method returns an array filled with every data collected since last call, e.g returns a list of 1000
    points after 1sec, 100 points after 0.1 sec etc...
    Due to this behavior, we cannot predict the length of the list. Thus, the time_vector is reconstructed in serial
    after collecting the results.
    """
    tinit = time()
    retrieved = self.sensor.get_stream()

    if self.mean:
      print 'before mean'
      retrieved = self.reshape(retrieved, self.mean)
      print 'after mean'

    nb_points = len(retrieved) if not self.mean else len(retrieved) / float(self.mean)
    time_vector = np.linspace(start=self.current_length, stop=self.current_length + 0.2, num=nb_points, endpoint=False)
    self.current_length += 0.2
    time_vector = np.around(time_vector, 5).tolist()
    return [time_vector, retrieved]

  def send(self, array):
    try:
      for output in self.outputs:
        output.send(array)
    except TimeoutError:
      raise
    except AttributeError:  # if no outputs
      raise

  def main(self):
    """
    Main loop of the streamer program.
    """
    try:
      self.sensor.start_stream()
      if type(self.sensor).__name__ == 'LabJack':
        time_vector_process = Process(target=self.time_vector)
        time_vector_process.start()
        while True:
          results = self.get_stream_from_labjack()
          array = OrderedDict(zip(self.labels, results))
          self.send(array)

      elif type(self.sensor).__name__ == 'OpenDAQ':
        while True:
          sleep(0.01)
          results = self.get_stream_from_opendaq()
          array = OrderedDict(zip(self.labels, results))
          self.send(array)

    except KeyboardInterrupt:
      self.sensor.close_streamer()
      pass
    except Exception:
      self.sensor.close_streamer()
      raise
