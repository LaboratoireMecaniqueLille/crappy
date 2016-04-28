# coding: utf-8
import serial

class Agilent34420ASensor(object):
	"""Sensor class for Agilent34420A devices."""
	def __init__(self,mode="VOLT",device='/dev/ttyUSB0',baudrate=9600,timeout=10):
		"""This class contains method to measure values of resistance or \
		tension on Agilent34420A devices. May work for other devices too, but \
		not tested.
		
		If you have issues with this class returning a lot of 'bad serial', \
		make sure you have the last version of pySerial.
		
		Parameters
		----------
		mode : {"VOLT","RES"} , default = "VOLT"
			Desired value to measure.
		device : str, default = '/dev/ttyUSB0'
			Path to the device.
		baudrate : int, default = 9600
			Desired baudrate.
		timeout : int or float, default = 10
			Timeout for the serial connection.
		"""
		self.device=device
		self.baudrate=baudrate
		self.timeout=timeout
		self.mode=mode
		self.ser = serial.Serial(port=self.device,baudrate=self.baudrate,timeout=self.timeout)
		self.ser.write("*RST;*CLS;*OPC?\n")
		self.ser.write("SENS:FUNC \""+self.mode+"\";  \n")
		self.ser.write("SENS:"+self.mode+":NPLC 2  \n")
		#ser.readline()
		self.ser.write("SYST:REM\n")
		self.getData()

	def getData(self):
		"""Read the signal, return False if error and print 'bad serial'."""
		try:
			self.ser.write("READ?  \n")
			#tmp = self.ser.readline()
			tmp = self.ser.read(self.ser.in_waiting)
			self.ser.flush()
			#print tmp
			return float(tmp)
		except:
			print "bad serial"
			#self.ser.read(self.ser.inWaiting())
			#print self.ser.inWaiting()
			#self.ser.flush()
			#time.sleep(0.5)
			return False
					
	def close(self):
		"""Close the serial port."""
		self.ser.close()
		
#def get_serial(ser):
  #try:
    #ser.write("READ?  \n")
    #tmp = ser.readline()
    #ser.flush()
    #print tmp
    #return float(tmp)
  #except:
    #print "bad serial"
    #return 9.99999999999999999999999999999


#ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
#ser.write("*RST;*CLS;*OPC?\n")
#ser.write("SENS:FUNC \"VOLT\";  \n")
#ser.write("SENS:VOLT:NPLC 1  \n")
#ser.readline()
#ser.write("SYST:REM\n")