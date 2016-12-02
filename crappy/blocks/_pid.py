# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup PID PID
# @{

## @file _pid.py
# @brief A proportional–integral–derivative controller (PID controller) is a control loop feedback mechanism.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _meta import MasterBlock
import time


class PID(MasterBlock):
    """
    A proportional–integral–derivative controller (PID controller) is a control loop feedback mechanism.
    PID controller continuously calculates an error value as the difference between a desired setpoint
    and a measured process variable.
    WORK IN PROGRESS.
    """

    def __init__(self, P, I, D, outMin=-10, outMax=10):
        super(PID, self).__init__()
        self.kp = P
        self.ki = I
        self.kd = D
        self.inAuto = False
        self.t_0 = time.time()
        self.outMin = outMin
        self.outMax = outMax

    # self.Iterm=0
    # self.mode='off'
    # self.lastOutput=0
    # def add_setpoint(self,link):
    # self.setpoint=link

    ##def setMode(mode):
    ##newMode=self.mode
    ##if newMode is 'on' or newMode is 'off':
    ##if newMode is not self.mode and newMode is 'on'
    ##self.initialize()
    ##self.inAuto=True
    ##else if newMode is 'off':
    ##self.inAuto=False

    # def initialize():
    # self.Iterm=self.lastOutput

    # if self.Iterm > outMax:
    # self.Iterm = outMax
    # else if self.Iterm < outMin:
    # self.Iterm=outMin

    # def compute()
    # if self.inAuto is True:
    # now=time.time()
    # timeChange=now-self.lastTime
    # self.input_=self.inputs[0].recv()
    # self.error=self.setpoint-self.input_
    # self.ki*=timeChange/self.lastTimeChange
    # self.kd/=timeChange/self.lastTimeChange
    # self.Iterm = self.ki * error*timeChange
    # if self.Iterm > outMax:
    # self.Iterm = outMax
    # else if self.Iterm < outMin:
    # self.Iterm=outMin
    # dInput=self.input_-self.lastInput
    # self.output=self.kp * error + self.Iterm - self.kd * dInput/timeChange

    # if self.output > outMax:
    # self.output = outMax
    # else if self.output < outMin:
    # self.output = outMin
    # self.lastOutput=self.output
    # Array=pd.DataFrame([[now-self.t_0, self.output]])
    # try:
    # for output in self.outputs:
    # output.send(Array)
    # except:
    # pass
    # self.lastInput = self.input_
    # self.lastTime = now
    # self.lastTimeChange=timeChange
    # return True
    # else return False




    # def main(self):
    # for input_ in self.inputs:
    # Sensor=self.inputs[0].recv()
    # t_init=time.time()-self.t0
    # self.lastTime=time.time()
    # while True:
    # self.compute()
    ##Data=pd.DataFrame()
    ##for input_ in self.inputs:
    ##Data=pd.concat([Data,self.consigne.recv()])
    ##Sensor=self.inputs[0].recv()
    ##[Series.last_valid_index][2]
