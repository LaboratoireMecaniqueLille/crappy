# -*- coding: utf-8 -*-
from _meta import MasterBlock
import time

class AutoDrive(MasterBlock):
    """Recieve position information and send motor commands calculated by the PID, via actuator CMdrive"""
    def __init__(self, actuator):
        self.actuator = actuator

    def main(self):
        try:
            while True:
                self.K = self.inputs[0].recv()
                if(self.K == ["Error"]):
                        raise Exception("Exception received from videoExtenso")
                self.coef = self.K['K']
                self.center = self.K['center']
                print "K in autodrive :",(self.coef), 'Center : ', self.center
                self.actuator.applyAbsoluteSpeed(int(self.coef))
        except (Exception,KeyboardInterrupt) as e:
            print "Exception in Autodrive : ", e
            try:
                self.actuator.applyAbsoluteSpeed(0)
            except:
                pass