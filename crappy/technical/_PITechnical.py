# coding: utf-8
import serial
from ._meta import motion
from ..sensor import PISensor
from ..actuator import PIActuator

class PI(motion.Motion):
    def __init__(self,port='/dev/ttyS0', timeout=1, baudrate=9600):
        """This class create an axis and opens the corresponding serial port.

        Parameters
        ----------
        port: str
                Path to the corresponding serial port, e.g '/dev/ttyS4'
        baud_rate : int, default = 38400
                Set the corresponding baud rate.
        timeout : int or float, default = 1
                Serial timeout.
        """
        self.port=port
        self.timeout=timeout
        self.baudrate = baudrate
        self.ser=serial.Serial(self.port,baudrate=self.baudrate,timeout=self.timeout)
        a=self.ser.write("%c%cSA%d\r"%(1,'0',10000)) #fixer acceleration de 10 000 a 100 000 microsteps par seconde au carre
        a=self.ser.write("%c%cSV%d\r"%(1,'0',10000)) # fixer vitesse

    def close(self):
        """Close the designated port"""
        self.ser.close()
    
    def stop(self):
        #TODO
        pass
    
    def clear_errors(self):
        #TODO
        pass
    
    def reset(self):
        #TODO
        pass