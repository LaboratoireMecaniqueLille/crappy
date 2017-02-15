from __future__ import print_function,division

from _measureByStep import MeasureByStep
from os import getpid
import time


# Inputs : a signal to put on the labjack
# Outputs : the signal, and the measures.

class ControlCommand(MeasureByStep):
  """
  Class to control and command from a single device. Works for labjack
  at the moment.
  """
  def __init__(self, device, *args, **kwargs):
    """
    Args :
      device: which device to control and command.
      verbose: prints the acquisition frequency if true.
      lua: used to control lua scripts. Specify a dictionary that
      will be initiated.
      The other arguments are the same as the MeasureByStep block.
    """
    self.verbose = kwargs.get('verbose', False)
    self.device = device
    self.lua = kwargs.get('lua', False)
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
      pass

    elif len(self.inputs) == 1 and len(self.outputs) == 0:
      self.main = self.main_command

    elif len(self.inputs) == 1 and len(self.outputs) > 0:
      if self.lua:
        self.main = self.main_control_command_lua
      else:
        self.main = self.main_control_command

      self.labels.append('signal')
      self.clear_results()
      if self.verbose:
        self.prepare_verbosity()
    else:
      raise TypeError('Wrong definition of ControlCommand block.')

  def main_command(self):

    while True:
      command = self.inputs[0].recv(blocking=True)['signal']
      self.device.set_cmd(command)

  def main_control_command(self):

    while True:
      t = time.time()
      data = self.acquire_data()
      command = self.inputs[0].recv(blocking=False)
      if command:
        signal = command['signal']
        self.device.set_cmd(signal)
      else:
        signal = 0.0
      data.append(signal)
      self.send(data)
      if self.verbose:
        self.increment_verbosity(data)
      if self.freq:
        delay = 1/self.freq-time.time()+t
        while delay > 0:
          time.sleep(delay/10)
          delay = 1/self.freq-time.time()+t

  def increment_verbosity(self, data):
    self.nb_acquisitions += 1
    time_interval = data[0] - self.elapsed
    if time_interval >= 1.:
      self.elapsed = data[0]
      self.queue.put(self.nb_acquisitions)

  def main_control_command_lua(self):

    while True:
      data = self.acquire_data()
      received = self.inputs[0].recv(blocking=False)

      if received:
        self.sensor.set_parameter_ram(received)
      data.append(0)
      self.send(data)

      if self.verbose:
        self.increment_verbosity(data)
