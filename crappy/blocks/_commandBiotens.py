# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup commandBiotens commandBiotens
# @{

## @file _commandBiotens.py
# @brief Receive a signal and send it for the Biotens actuator.
#
# @author Robin Siemiatkowski, Victor Couty
# @version 0.1
# @date 13/02/2017

from _masterblock import MasterBlock
import time

class CommandBiotens(MasterBlock):
  """Receive a signal and send it for the Biotens actuator"""

  def __init__(self, biotens_technicals, signal_label='signal', speed=5, **kwargs):
    """
    Receive a signal and translate it for the Biotens actuator.

    Args:
        biotens_technicals : list of crappy.technical.Biotens objects.
            List of all the axes to control.
        signal_label : str, default = 'signal'
            Label of the data to be transfered.
        speed: int, default = 5
            Wanted speed, in mm/min.
    """
    self.labels = kwargs.get("labels")
    if not self.labels:
      self.labels = ['t(s)']
      for i in range(len(biotens_technicals)):
        self.labels.append('position%d'%(i+1))
    MasterBlock.__init__(self)
    self.biotens_technicals = biotens_technicals
    self.speed = speed
    self.signal_label = signal_label
    for biotens_technical in self.biotens_technicals:
      biotens_technical.clear_errors()

  def main(self):
    try:
      # print "top command"
      last_cmd = 0
      self.last_time = self.t0
      while True:
        Data = self.inputs[0].recv()
        # try:
        # cmd=Data['signal'].values[0]
        # except AttributeError:
        cmd = Data[self.signal_label]
        if cmd != last_cmd:
          for biotens_technical in self.biotens_technicals:
            biotens_technical.actuator.set_speed(cmd * self.speed)
          last_cmd = cmd
        t = time.time()
        if (t - self.last_time) >= 0.2:
          # print "top command3"
          self.last_time = t
          data = [t-self.t0]
          for biotens_technical in self.biotens_technicals:
            data.append(biotens_technical.sensor.get_position())
          self.send(data)

    except (Exception, KeyboardInterrupt) as e:
      print "Exception in CommandBiotens : ", e
      for biotens_technical in self.biotens_technicals:
        biotens_technical.actuator.stop_motor()
        # raise
