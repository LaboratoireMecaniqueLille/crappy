# -*- coding: utf-8 -*-
from _meta import MasterBlock
import time
from sys import stdout

class AutoDrive(MasterBlock):
    """Recieve position information and send motor commands calculated by the PID, via actuator CMdrive"""
    def __init__(self, technical):
        self.technical = technical

    def main(self):
        try:
            while True:
                self.K = self.inputs[0].recv()
                if(self.K == ["Error"]):
                        raise Exception("Exception received from videoExtenso")
                self.coef = self.K['K']
                self.center = self.K['center']
                stdout.write("\rK in autodrive :{}, Center : {}".format(self.coef, self.center))
                stdout.flush()
                self.technical.actuator.set_speed(int(self.coef))
        except (Exception,KeyboardInterrupt) as e:
            print "Exception in Autodrive : ", e
            try:
                self.technical.actuator.set_speed(0)
            except:
                pass