# coding: utf-8
import serial
import time

from crappy2.technical._meta import motion
from ..sensor import OrientalSensor
from ..actuator import OrientalActuator


class Oriental(motion.Motion):
    """Open both a BiotensSensor and BiotensActuator instances."""
    def __init__(self, baudrate=115200, port='/dev/ttyUSB0', num_device=1):
        """
        Open the connection, and initialise the Biotens.

        You should always use this Class to communicate with the Biotens.

        Parameters
        ----------
        port : str, default = '/dev/ttyUSB0'
            Path to the correct serial ser.
        size : int of float, default = 30
            Initial size of your test sample, in mm.
        """
        super(Oriental, self).__init__(port, baudrate)
        self.baudrate = baudrate
        self.num_device = num_device
        self.port = port
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1)
        self.sensor = OrientalSensor(ser=self.ser, num_device=self.num_device)
        self.actuator = OrientalActuator(ser=self.ser, num_device=self.num_device)

    def write_cmd(self, cmd):
        self.ser.write("{0}\n".format(cmd))
        ret = self.ser.readline()
        # while ret != '{0}>'.format(self.num_device):
        while ret != '' and ret != '{0}>'.format(self.num_device):
            print ret
            ret = self.ser.readline()

    def clear_errors(self):
        self.write_cmd("ALMCLR")

    def close(self):
        self.stop()
        self.ser.close()

    def stop(self):
        self.write_cmd("SSTOP")

    def reset(self):
        self.clear_errors()
        self.write_cmd("RESET")
        self.write_cmd("TALK{}".format(self.num_device))
        self.clear_errors()
