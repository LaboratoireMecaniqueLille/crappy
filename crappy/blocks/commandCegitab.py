#!/usr/bin/python
# -*- coding: utf-8 -*-
# Cegitab serial ports

from .masterblock import MasterBlock
import time
from collections import OrderedDict
import math


# Désignation		Win	Port name	Baudrate	ByteSize	Parity
# Actionneur vérin	COM5	'/dev/ttyUSB0'	19200
# Capteur effort		COM4	'dev/ttyS0'	9600		7		Odd
# Platine XY - X		COM4	'/dev/ttyUSB1'	19200
# Platine XY - Y		COM3	'/dev/ttyUSB2'	19200


class PipeCegitab(MasterBlock):
    def __init__(self, Actuator, Load,MaxLoad):
        super(PipeCegitab, self).__init__()
        self.Actuator = Actuator
        self.Load = Load
        self.MaxLoad = MaxLoad

    def loop(self):
        time.sleep(0.1)
        #pos = self.Actuator.Position()
        pos = self.Actuator.Position() #in dummy mode. Line above for IRL
        #F = self.Load.Read()
        #r, F = self.Load.get_data() #using dummy_sensor
        F = pos*40 #F is a function of pos (close enough from a real material).
        #LOAD LIMIT ----> #A AMELIORER
        if math.fabs(F)>self.MaxLoad:
            print "OVERLOAD"
            #speedback=self.Actuator.get_speed()
            #print speedback
            #self.Actuator.set_speed(-speedback)
            self.Actuator.set_speed(0)
        # Array=pd.DataFrame([[time.time()-self.t0,pos,F]],columns=['t(s)','position(mm)','F(N)'])
        Array = OrderedDict(zip(['t(s)', 'position(mm)', 'F(N)'], [time.time() - self.t0, pos, F]))
        #print Array
        self.send(Array)
