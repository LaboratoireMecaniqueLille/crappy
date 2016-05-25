# coding: utf-8
import serial
from ._meta import motion

class BiaxeSensor(motion.MotionSensor):
    """Declare a new axis for the Biaxe"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=38400, timeout=1, ser=None):
        """This class create an axis and opens the corresponding serial port.
        
        Parameters
        ----------
        port : str
                Path to the corresponding serial port, e.g '/dev/ttyS4'
        baudrate : int, default = 38400
                Set the corresponding baud rate.
        timeout : int or float, default = 1
                Serial timeout.
        """
        self.port = port
        self.baudrate= baudrate
        self.timeout=timeout
        if ser != None:
            self.ser = ser
        else:
            self.ser=serial.Serial(self.port,self.baudrate,
                                        serial.EIGHTBITS,serial.PARITY_EVEN
                                        ,serial.STOPBITS_ONE,self.timeout)
            self.ser.write("OPMODE 0\r\n EN\r\n")
        
    def get_position(self):
        """
        TODO
        Search for the physical position of the motor
        """
        pass