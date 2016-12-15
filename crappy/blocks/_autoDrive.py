# -*- coding: utf-8 -*-
##  @addtogroup blocks
# @{

##  @defgroup AutoDrive AutoDrive
# @{

## @file _autoDrive.py
# @brief Recieve position information and send motor commands calculated by the PID, via actuator CMdrive.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 05/07/2016

from _masterblock import MasterBlock
import time
from sys import stdout


class AutoDrive(MasterBlock):
    """
    Recieve position information and send motor commands calculated by the PID, via actuator CMdrive
    """

    def __init__(self, technical):
        """
        Args:
            technical: technical object (Motion) which received the command.
        """
        super(AutoDrive, self).__init__()
        self.technical = technical
        self.K = None
        self.coef = None
        self.center = None

    def main(self):
        """
        Apply the command received by a link to the technical object.
        """
        try:
            while True:
                self.K = self.inputs[0].recv()
                if self.K == ["Error"]:
                    raise Exception("Exception received from videoExtenso")
                self.coef = self.K['K']
                self.center = self.K['center']
                stdout.write("\rK in autodrive :{}, Center : {}".format(self.coef, self.center))
                stdout.flush()
                self.technical.actuator.set_speed(int(self.coef))
        except (Exception, KeyboardInterrupt) as e:
            print "Exception in Autodrive : ", e
            try:
                self.technical.actuator.set_speed(0)
            except Exception as e:
                print e
