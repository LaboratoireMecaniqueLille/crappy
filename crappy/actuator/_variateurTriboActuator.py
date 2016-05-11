#!/usr/bin/python
# -*- coding: utf-8 -*-

import serial
import time
class VariateurTriboActuator(object):
	def __init__(self,ser_servostar,ser_arduino):
		self.ser_servostar=ser_servostar
		self.ser_arduino=ser_arduino
		self.init=False
	def stop_motor(self):
		self.ser_servostar.write('dis\r\n')
		
	def set_mode_position(self):
		self.ser_servostar.write('opmode 8\r\n')
		self.mode='position'
		time.sleep(0.1)
		print self.mode
		
	def go_position(self,position,speed=20000,acc=200,dec=200):
		self.ser_servostar.write("ORDER 0 "+str(position)+" "+
		str(speed)+" 8192 "+str(acc)+" "+str(dec)+" "+" 0 0 0 0\r\n") #creating the order for the motor example ORDER 0 1000 20000 8192 200 200 0 0 0 0\r\n  
		self.ser_servostar.write("MOVE 0\r\n")                   #activates the order
		
	def set_mode_analog(self):
		self.ser_servostar.write('opmode 1\r\n')
		self.ser_servostar.write('ancnfg 0\r\n')
		time.sleep(0.1)
		self.mode='effort'
		print self.mode
		
	def go_effort(self,effort):
		print str(effort)
		self.ser_arduino.write(str(effort))
		
	def initialisation(self):
		self.ser_servostar.write('opmode 8\r\n')
		self.ser_servostar.write('en\r\n')
		self.ser_servostar.write('mh\r\n')
		self.init=True
		
	
