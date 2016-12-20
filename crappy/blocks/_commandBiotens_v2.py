from _masterblock import MasterBlock
import time
# import pandas as pd
from collections import OrderedDict


##  @addtogroup blocks
# @{

##  @defgroup CommandBiotens_v2 CommandBiotens_v2
# @{

## @file _commandBiotens_v2.py
# @brief Receive a signal and translate it for the Biotens actuator.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016


class CommandBiotens_v2(MasterBlock):
  """Receive a signal and translate it for the Biotens actuator"""

  def __init__(self, biotens_technicals, labels='None', speed=5):
    """
    Receive a signal and translate it for the Biotens actuator.

    CommandBiotens(biotens_technical,speed=5)

    Args:
        biotens_technicals : list of crappy.technical.Biotens object.

        speed: int, default = 5
    """
    super(CommandBiotens_v2, self).__init__()
    self.biotens_technicals = biotens_technicals
    self.speed = speed
    self.labels = labels

    for biotens_technical in self.biotens_technicals:
      biotens_technical.actuator.clear_errors()

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
        cmd = Data['signal']
        if cmd != last_cmd:
          for biotens_technical in self.biotens_technicals:
            biotens_technical.actuator.setmode_speed(cmd * self.speed)
          last_cmd = cmd
        ##print "top command2"
        # Data=self.inputs[0].recv(blocking=False)
        # try :
        # cmd=Data['signal'].values[0]
        ##print "cmd : ", cmd
        # if cmd!= last_cmd:
        # for biotens_technical in self.biotens_technicals:
        # biotens_technical.actuator.setmode_speed(cmd*self.speed)
        # last_cmd=cmd
        # except TypeError:
        # pass
        t = time.time()
        data = [t - self.t0]
        if (t - self.last_time) >= 0.2:
          # print "top command3"
          self.last_time = t
          # data=[t-self.t0]
          for biotens_technical in self.biotens_technicals:
            position_ = biotens_technical.sensor.read_position()
            data.append(position_)
          # Array=pd.DataFrame([[t-self.t0,position]],columns=['t(s)','position'])
          Array = OrderedDict(zip(self.labels, data))
          try:
            for output in self.outputs:
              # print "sending position ..."
              output.send(Array)
              # print "position sent ..."
          except:
            # print "no outputs"
            pass
    except (Exception, KeyboardInterrupt) as e:
      print "Exception in CommandBiotens : ", e
      for biotens_technical in self.biotens_technicals:
        biotens_technical.actuator.stop_motor()
      raise
