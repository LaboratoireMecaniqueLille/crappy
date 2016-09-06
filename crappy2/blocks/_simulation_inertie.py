# -*- coding:utf-8 -*-
import Tix
from Tkinter import *
import os
import time
import tkFont
import tkFileDialog
from _meta import MasterBlock
import Tkinter
import tkMessageBox
import threading
import math

from _meta import MasterBlock

class SimulationInertie(MasterBlock):
    def __init__(self, VariateurTribo, labjack,labjack_hydrau,conditioner,Vmax,Vmin,F,inertie,cycle):
	self.VariateurTribo=VariateurTribo
	MasterBlock.__init__(self)
	self.labjack = labjack
        self.labjack_hydrau=labjack_hydrau
        self.labjack_hydrau.set_cmd(0)
        self.labjack.set_cmd_ram(0, 46002)  # sets the pid off
        self.labjack.set_cmd_ram(-41, 46000)  # sets the setpoint at 0 newton
	self.conditioner = conditioner
	self.MaxSpeed = Vmax
	self.MinSpeed = Vin
	self.Force = F
	self.SimulatedInertia = inertie
	self.cycleNumber = cycle
	
    def main(self):
	for i in range(len(self.conditioner)):
	    self.conditioner[i].reset()

	self.outputs[2].send(1)
        try:
            cycle = self.cycleNumber.get()
            while cycle > 0:
                self.outputs[0].send(1)
                tStart = time.time()
                tAfter = time.time()


                while tAfter - tStart < 0.5 and not self.experiment._stopevent.isSet():

                    value = self.inputs[0].recv()

                    tAfter = time.time()

		print("Acceleration")

                while float(value['Vit']) <= self.MaxSpeed * 95. / 100. :

                    self.labjack.set_cmd(self.MaxSpeed)
                    value = self.inputs[0].recv()

                tStart = time.time()
                tAfter = time.time()
		
		print("Stabilisation")
                while tAfter - tStart < 2 :
                    value = self.inputs[0].recv()
                    tAfter = time.time()

                tStart = time.time()
                tAfter = time.time()


                C0 = float(value['Couple'])
                V0 = float(value['Vit'])
                V1 = V0

                self.labjack.set_cmd_ram(1,46002)
		self.labjack_hydrau.set_cmd(10)
                self.VariateurTribo.actuator.set_mode_analog()
                self.labjack.set_cmd_ram(int(self.Force)-41,46000)
		
		print("Freinage")
		self.Informations.configure(bg="red")
                while V1 > self.MinSpeed :  # Freinage
                    tStart = tAfter
                    tAfter = time.time()
                    deltaVit = (tAfter - tStart)*60 * (float(value['Couple']) - C0) /(2*math.pi *float(self.SimulatedInertia))
                    #print 'delta', deltaVit

                    V1 = V1 - deltaVit
                    self.labjack.set_cmd(V1)
                    # self.comediOut.set_cmd(V1)
                    value = self.inputs[0].recv()
		
		self.labjack_hydrau.set_cmd(0)
                self.VariateurTribo.actuator.set_mode_position()
                self.VariateurTribo.actuator.go_position(0)
                self.labjack.set_cmd_ram(0, 46002)
                self.labjack.set_cmd_ram(-41, 46000)
                # self.comediOut.set_cmd(0)
                self.outputs[0].send(0)

                cycle -= 1
                if cycle>0:
		  tStart = time.time()
		  tAfter = time.time()

		  while tAfter - tStart < 1 :
		      tAfter = time.time()
	    tStart = time.time()
	    tAfter = time.time()
	    while tAfter - tStart < 1 :
                tAfter = time.time()
            self.labjack_hydrau.set_cmd(0)
	    self.labjack.set_cmd(0)
	    self.outputs[2].send(0)
