# coding: utf-8
import serial
from ._meta import motion
from ..actuator import BiaxeActuator
from ..sensor import BiaxeSensor


class Biaxe(motion.Motion):
    """Declare a new axis for the Biaxe"""

    def __init__(self, port='/dev/ttyUSB0', baudrate=38400, timeout=1):
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
        super(Biaxe, self).__init__(port, baudrate)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = serial.Serial(self.port, self.baudrate,
                                 serial.EIGHTBITS, serial.PARITY_EVEN
                                 , serial.STOPBITS_ONE, self.timeout)
        self.ser.write("OPMODE 0\r\n EN\r\n")
        self.sensor = BiaxeSensor(ser=self.ser)
        self.actuator = BiaxeActuator(ser=self.ser)

    def stop(self):
        self.ser.write("J 0\r\n")

    def reset(self):
        # TODO
        pass

    def close(self):
        """Close the designated port"""
        self.actuator.set_speed(0)
        self.stop()
        self.ser.close()

    def clear_errors(self):
        """Reset errors"""
        self.ser.write("CLRFAULT\r\n")
        self.ser.write("OPMODE 0\r\n EN\r\n")
