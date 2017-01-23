from __future__ import print_function

from _measureByStep import MeasureByStep
from os import getpid
import time


# Inputs : a signal to put on the labjack
# Outputs : the signal, and the measures.

class ControlCommand(MeasureByStep):
  def __init__(self, device, *args, **kwargs):
    self.verbose = kwargs.get('verbose', False)
    self.device = device
    MeasureByStep.__init__(self, sensor=self.device, *args, **kwargs)

  def vprint(*args):
    """
    Function used in case of verbosity.
    """
    print('[crappy.blocks.ControlCommand] PID', getpid(), *args)

  def prepare(self):
    MeasureByStep.prepare(self)
    print(self.inputs, self.outputs)
    if len(self.inputs) == 0 and len(self.outputs) > 0:
      pass  # because this is already the measurebystep.main.
    elif len(self.inputs) == 1 and len(self.outputs) == 0:
      self.main = self.main_command
    elif len(self.inputs) == 1 and len(self.outputs) > 0:
      self.main = self.main_control_command
      self.labels.append('signal')
      self.clear_results()
      if self.verbose:
        self.prepare_verbosity()
    else:
      raise Exception('Wrong definition of ControlCommand block.')

  def main_command(self):

    while True:
      command = self.inputs[0].recv()['signal']
      self.device.set_cmd(command)

  def main_control_command(self):

    while True:

      data = self.acquire_data()
      command = self.inputs[0].recv()['signal']
      data.append(command)
      self.send_to_compacter(data)
      self.device.set_cmd(command)

      if self.verbose:
        self.nb_acquisitions += 1
        time_interval = data[0] - self.elapsed
        if time_interval >= 1.:
          self.elapsed = data[0]
          self.queue.put(self.nb_acquisitions)
