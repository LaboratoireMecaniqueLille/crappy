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
from time import sleep


class Streamer(MasterBlock):
  """
  Send a stream of data.
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
        mean : int, number to shrink data. For instance, if 10 000 value are in input, averaging = 10 will send
        in output 1000 values.

    This block does the time vector reconstruction, then assemble it with the results matrix read from the
    LJM Buffer.
    """
    super(Streamer, self).__init__()
    assert sensor, "No input sensor defined."
    self.sensor = sensor
    self.labels = labels if labels else ['time(sec)'] + self.sensor.channels
    self.averaging = kwargs.get('mean', None)

    if type(self.sensor).__name__ == 'LabJack':
      global queue
      queue = Queue(2)
    elif type(self.sensor).__name__ == 'OpenDAQ':
      self.current_length = 0.

  def time_vector(self):
    """
    Time vector reconstruction. This has to be made in another process. Creates the time vector of the
    length = nb_samples, then puts it in a queue to be retrieved as soon as the result matrix pops.
    """
    sample_number = 0
    # print "Streamer / time vector reconstruction: PID", getpid()
    while True:
      try:
        ratio = self.sensor.scans_per_read / float(self.sensor.scan_rate_per_channel)
        nb_points = self.sensor.scans_per_read if not self.averaging else self.sensor.scans_per_read / self.averaging
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
    reshaped = []
    for length in xrange(np.shape(nparray)[1] / n):
      reshaped.append(np.mean(nparray[:, length * n: int((length + 1 - 1 / float(n)) * n)], axis=1).tolist())
    return np.array(reshaped).transpose()

  def get_stream_from_labjack(self):
    results = []
    retrieved = self.sensor.get_stream()[1]
    deinterlaced = np.array([retrieved[each::self.sensor.nb_channels]
                             for each in xrange(self.sensor.nb_channels)])
    if self.averaging:
      deinterlaced = self.reshape(deinterlaced, self.averaging)

    for each in xrange(self.sensor.nb_channels):
      liste_temp = self.sensor.gain[each] * deinterlaced[each] + self.sensor.offset[each]
      results.append(liste_temp.tolist())
    results.insert(0, queue.get())
    return results

  def get_stream_from_opendaq(self):
    retrieved = self.sensor.get_stream()
    if self.averaging:
      retrieved = self.reshape(retrieved, self.averaging)
    nb_points = len(retrieved) if not self.averaging else len(retrieved) / self.averaging
    time_vector = np.linspace(start=self.current_length, stop=self.current_length + nb_points / 1000., num=nb_points,
                              endpoint=False)
    self.current_length += nb_points / 1000.
    time_vector = np.around(time_vector, 5).tolist()
    # print 'time vector:', time_vector[:5], time_vector[-5:]
    # print 'length time vector: %d length retrieved: %d', len(time_vector), len(retrieved)
    return [time_vector[:5], retrieved[:5]]

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
          sleep(0.1)
          results = self.get_stream_from_opendaq()
          array = OrderedDict(zip(self.labels, results))
          self.send(array)


    except KeyboardInterrupt:
      pass
    except Exception:
      raise
