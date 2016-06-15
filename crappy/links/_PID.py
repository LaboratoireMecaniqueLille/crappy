# coding: utf-8
from ._metaCondition import MetaCondition
import time


class PID(MetaCondition):
    """WIP, not working yet."""

    def __init__(self, P, I, D, label_consigne, label_retour, label_consigne_sortie=None, consigne=None, outMin=-10,
                 outMax=10, add_current_value=True):
        self.P = P
        self.I = I
        self.D = D
        self.label_consigne = label_consigne
        self.label_consigne_sortie = label_consigne_sortie
        if self.label_consigne_sortie is None:
            self.label_consigne_sortie = label_consigne
        self.label_retour = label_retour
        self.outMin = outMin
        self.outMax = outMax
        self.first = True
        self.consigne = consigne
        self.add_current_value = add_current_value

    def evaluate(self, value):
        try:
            value.update(self.external_trigger.recv())
        except AttributeError:  # no external trigger
            pass
        self.retour = value[self.label_retour]
        if self.consigne is None:
            self.consigne = value[self.label_consigne]
        if self.first:
            self.lastTime = time.time()
            self.last_retour = self.retour
            self.first = False
            self.lastTimeChange = 10 ** 118  # for initialization
        # print "recv : ",self.retour
        self.compute()
        # print "output : ", self.output
        if self.add_current_value:
            val = self.output + self.retour
        else:
            val = self.output
        if val > self.outMax:
            val = self.outMax
        elif val < self.outMin:
            val = self.outMin
        try:
            value.pop(self.label_consigne)
        except KeyError:
            pass
        value[self.label_consigne_sortie] = val
        # print "send : ",val
        return value

    # def initialize(self):
    # self.Iterm=self.lastOutput

    # if self.Iterm > outMax:
    # self.Iterm = outMax
    # elif self.Iterm < outMin:
    # self.Iterm=outMin

    def compute(self):
        # if self.inAuto is True:
        now = time.time()
        timeChange = now - self.lastTime
        # self.input_=self.inputs[0].recv()
        self.error = self.consigne - self.retour
        self.I *= timeChange / self.lastTimeChange
        self.D /= timeChange / self.lastTimeChange
        self.Iterm = self.I * self.error * timeChange
        if self.Iterm > self.outMax:
            self.Iterm = self.outMax
        elif self.Iterm < self.outMin:
            self.Iterm = self.outMin
        dInput = self.retour - self.last_retour
        self.output = self.P * self.error + self.Iterm + self.D * dInput / timeChange
        if self.output > self.outMax:
            self.output = self.outMax
        elif self.output < self.outMin:
            self.output = self.outMin
        self.lastOutput = self.output
        self.last_retour = self.retour
        self.lastTime = now
        self.lastTimeChange = timeChange

        # def setMode(mode):
        # newMode=self.mode
        # if newMode is 'on' or newMode is 'off':
        # if newMode is not self.mode and newMode is 'on'
        # self.initialize()
        # self.inAuto=True
        # else if newMode is 'off':
        # self.inAuto=False

        # def main(self):
        # for input_ in self.inputs:
        # Sensor=self.inputs[0].recv()
        # t_init=time.time()-self.t0
        # self.lastTime=time.time()
        # while True:
        # self.compute()
        # Data=pd.DataFrame()
        # for input_ in self.inputs:
        # Data=pd.concat([Data,self.consigne.recv()])
        # Sensor=self.inputs[0].recv()
        # [Series.last_valid_index][2]

        # value[self.label_consigne]=val*self.coeff
        # return value
