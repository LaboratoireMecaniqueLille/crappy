# coding: utf-8
import serial
import time
#import os

class CmDrive():
	""" Open a new default serial port for communication with Servostar"""
	def __init__(self):	 
		self.myPort = '/dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0' 
		self.baudrate = 9600
		self.ser = serial.Serial(self.myPort, self.baudrate)
		pass
   
	def setConnection(self, port, baudrate):
		"""Open a new specified serial port for communication with Servostar"""
		self.myPort = port 
		self.baudrate = baudrate
		self.ser = serial.Serial(self.myPort, self.baudrate)
		self.ser.close()
		return self.ser
		
	"""Methods controlling motion
	============================="""
	"""Stop the motor motion"""
	def stopMotion(self):
		self.ser.close()#close serial connection before to avoid errors
		self.ser.open()
		self.ser.write('SL 0\r')
		self.ser.readline()
		self.ser.close()
		
	"""Absolut displacement from zero"""
	def applyAbsoluteMotion(self,position):
		self.ser.close() #close serial connection before to avoid errors
		self.ser.open() # open serial port
		self.ser.write('MA %i\r' %(position)) #send ASCII characters to apply the selected motion task
		self.ser.readline()
		self.ser.close()#close serial connection
		
	"""Relative displacement from current position"""
	def applyRelativeMotion(self,num):
		self.ser.close() #close serial connection before to avoid errors
		self.ser.open() # open serial port
		self.ser.write('MR %i\r' %(num)) #send ASCII characters to apply the selected motion task
		self.ser.readline()
		self.ser.close()#close serial connection
	
	"""Methods controlling speed
	============================"""
	"""Positive displacement at a setted speed"""	
	def applyPositiveSpeed(self, speed):
		self.ser.close()#close serial connection before to avoid errors
		self.ser.open()# open serial port
		#velocity = input ('Velocity: \n') #request to the user about velocity
		if speed < 1000000:
			self.ser.write('SL %i\r' % speed) # send ASCII characters to the servostar to apply velocity task
			self.ser.readline()
		else:
			print 'Maximum speed exeeded'
		self.ser.close()#close serial connection
		
	"""Negative displacement at a setted speed"""		 
	def applyNegativeSpeed(self, speed):
		self.ser.close()#close serial connection before to avoid errors
		self.ser.open()# open serial port
		#velocity = input ('Velocity: \n')#request to the user about velocity
		if speed < 1000000:
			self.ser.write('SL -%i\r' % speed)# send ASCII characters to the servostar to apply velocity task
			self.ser.readline()
		else:
			print 'Maximum speed exeeded'
		self.ser.close()#close serial connection

	"""Positive or Negative displacement at a setted speed"""		 
	def applyAbsoluteSpeed(self, speed):
		self.ser.close()#close serial connection before to avoid errors
		self.ser.open()# open serial port
		#velocity = input ('Velocity: \n')#request to the user about velocity
		if abs(speed) < 1000000:
			self.ser.write('SL '+str(int(speed))+'\r')# send ASCII characters to the servostar to apply velocity task
			self.ser.read(self.ser.inWaiting())
		else:
			print 'Maximum speed exeeded'
		self.ser.close()#close serial connection
	
	"""Methods checking & controlling position
	=========================================="""
	"""Search for the physical position of the motor"""
	def examineLocation(self):
		self.ser.close()
		ser=self.setConnection(self.myPort, self.baudrate) # initialise serial port
		self.ser.open()
		self.ser.write('PR P \r') # send 'PFB' ASCII characters to request the location of the motor
		pfb = self.ser.readline() # read serial data from the buffer
		pfb1 = self.ser.readline()# read serial data from the buffer
		print '%s %i' %(pfb,(int(pfb1))) #print location
		print '\n' 
		self.ser.close() #close serial connection
		return int(pfb1)
	
	"""Reset the serial communication, before reopen it to set displacement to zero"""
	def resetZero(self):
		self.ser.close() #????????????
		self.ser=self.setConnection(self.myPort, self.baudrate) # initialise serial port
		self.ser.open() # open serial port
		import Tkinter
		import tkMessageBox
		result = tkMessageBox.askyesno('resetZero', 'Warning! The recorded trajectories will be erased, continue?')#send request to the user if he would reset the system
		if result is True: 
			self.ser.write('DIS\r') # send 'DIS' ASCII characters to disable the motor
			self.ser.write('SAVE\r')# send 'SAVE' ASCII characters to SAVE servostar values
			self.ser.write('COLDSTART\r')# send 'COLDSTART' ASCII characters to reboot servostar
			k=0
			#print different stages of booting
			while k<24:
				print self.ser.readline()
				k+=1
			#self.ser.close() #close serial connection
			return 1
		else:
			#self.ser.close() #close serial connection
			return 0

	"""Reset the position to zero"""
	def moveZero(self):
		self.ser.open() # open serial port
		self.ser.write('MA 0\r')# send 'MH' ASCII characters for requesting to the motor to return at zero position
		self.ser.readline()
		self.ser.close()#close serial connection

	def close_port(self):
		"""Close the designated port"""
		self.ser.close()
	
	def CLRFAULT(self):
		"""Reset errors"""
		self.ser.write("CLRFAULT\r\n")
		self.ser.write("OPMODE 0\r\n EN\r\n")

