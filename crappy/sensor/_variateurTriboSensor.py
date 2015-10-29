#!/usr/bin/python
# -*- coding: utf-8 -*-

import serial

class VariateurTriboSensor(object):
	def __init__(self,ser):
		self.ser_servostar=ser
	
	def read_position(self): 
		while self.ser_servostar.inWaiting() > 0:
			self.ser_servostar.read(1)
		out= ''
		self.ser_servostar.write('pfb\r\n')
		time.sleep(0.1)
		out = ''
		while self.ser_servostar.inWaiting() > 0:
			out += self.ser_servostar.read(1)
		if out != '':
			datastring=out.split('\r')
			if 'PFB' in datastring[0]:
				datastring2=datastring[1].split('\n')
				position=int(datastring2[1])
				#print position
				datastring=''
				datastring2=''
		
		return position