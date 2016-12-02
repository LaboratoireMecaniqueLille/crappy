# coding: utf-8
from _meta import MasterBlock
##  @addtogroup blocks
# @{

##  @defgroup CommandBiaxe CommandBiaxess
# @{

## @file _autoDrive.py
# @brief Receive a signal and translate it for the Biaxe actuator.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016


class CommandBiaxe(MasterBlock):
    """Receive a signal and send it for the Biaxe actuator"""

    def __init__(self, biaxe_technicals, signal_label='signal', speed=500):
        """
        Receive a signal and translate it for the Biaxe actuator.

        Args:
            biaxe_technicals : list of crappy.technical.Biaxe objects
                List of all the axes to control.
            signal_label : str, default = 'signal'
                Label of the data to be transfered.
            speed: int, default = 500
                Wanted speed. 1 is equivalent to a speed of 0.002 mm/s.
        """
        super(CommandBiaxe, self).__init__()
        self.biaxe_technicals = biaxe_technicals
        self.speed = speed
        self.signal_label = signal_label
        for biaxe_technical in self.biaxe_technicals:
            biaxe_technical.actuator.new()

    def main(self):
        try:
            last_cmd = 0
            while True:
                Data = self.inputs[0].recv()
                # try:
                # cmd=Data['signal'].values[0]
                # except AttributeError:
                cmd = Data[self.signal_label]
                if cmd != last_cmd:
                    if last_cmd*cmd < 0:
                        for biaxe_technical in self.biaxe_technicals:
                            biaxe_technical.actuator.set_speed(0)
                    for biaxe_technical in self.biaxe_technicals:
                        biaxe_technical.actuator.set_speed(cmd * self.speed)
                    last_cmd = cmd
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in CommandBiaxe : ", e
            for biaxe_technical in self.biaxe_technicals:
                biaxe_technical.actuator.set_speed(0)
                biaxe_technical.close()
                # raise
