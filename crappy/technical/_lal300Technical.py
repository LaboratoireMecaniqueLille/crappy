import time
import serial
from serial import SerialException
from ..sensor import SensorLal300
from ..actuator import ActuatorLal300

class TechnicalLal300(object):
	
    def __init__(self,param):
        self.param=param
        self.ser=serial.Serial(port=param['port'], #Configuration du port serie à l'aide de PySerial
        baudrate=param['baudrate'],
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=param['timeout'],
        rtscts=False,
        write_timeout=None,
        dsrdtr=False,
        inter_byte_timeout=None)
        self.actuator=ActuatorLal300(self.param,self.ser) #Appel de la sous-classe ActuatorLal300 avec les parametres situes dans le programme lal300Main.py
        self.sensor=SensorLal300(self.param,self.ser)  #Appel de la sous-classe SensorLal300 avec les parametres situes dans le programme lal300Main.py