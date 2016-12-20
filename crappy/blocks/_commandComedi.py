# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup CommandComedi CommandComedi
# @{

## @file _commandComedi.py
# @brief Receive a signal and send it to a Comedi card.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _masterblock import MasterBlock
import time
import os


# import gc

class CommandComedi(MasterBlock):
  """Receive a signal and send it to a Comedi card"""

  def __init__(self, comedi_actuators, signal_label="signal"):
    """
    Receive a signal and translate it for the Comedi card.

    CommandComedi(comedi_actuators)

    Args:
        comedi_actuators : list of crappy.actuators.ComediActuator objects.
            List of all the outputs to control.
        signal_label : str, default = 'signal'
            Label of the data to be transfered.
    """
    super(CommandComedi, self).__init__()
    self.signal_label = signal_label
    self.comedi_actuators = comedi_actuators
    if type(self.signal_label) == str:
      self.signal_label = [self.signal_label] * len(self.comedi_actuators)

    print "command comedi! "

  def main(self):
    print "command comedi! :", os.getpid()
    try:
      # try:
      # for comedi_actuator in self.comedi_actuators:
      # comedi_actuator.open_handle()
      # except Exception as e:
      # print e
      last_cmd = [0] * len(self.comedi_actuators)
      while True:
        cmd = []
        Data = self.inputs[0].recv()
        # try:
        # cmd=Data['signal'].values[0]
        # except AttributeError:
        for label in self.signal_label:
          cmd.append(Data[label])
        # print cmd
        # if cmd!= last_cmd:
        for i, comedi_actuator in enumerate(self.comedi_actuators):
          if cmd[i] != last_cmd[i]:
            comedi_actuator.set_cmd(cmd[i])
          # print cmd[i]
          last_cmd[i] = cmd[i]
          # print cmd
    except (Exception, KeyboardInterrupt) as e:
      print "Exception in CommandComedi : ", e
      for comedi_actuator in self.comedi_actuators:
        comedi_actuator.close()
        # raise
