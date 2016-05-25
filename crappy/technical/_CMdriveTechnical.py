# coding: utf-8
import serial
import time
from ._meta import motion
from ..actuator import CmDriveActuator
from ..sensor import CmDriveSensor

class CmDrive(motion.Motion):
    """ Open a new default serial port for communication with Servostar"""
    def __init__(self, port='/dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0', baudrate=9600):	 
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate)
        self.sensor = CmDriveSensor(self.ser)
        self.actuator = CmDriveActuator(self.ser)

    """Stop the motor motion"""
    def stop(self):
            self.ser.close()#close serial connection before to avoid errors
            self.ser.open()
            self.ser.write('SL 0\r')
            #self.ser.readline()
            self.ser.close()
    
    """Reset the serial communication, before reopen it to set displacement to zero"""
    def reset(self):
            self.ser.close()
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

    def close(self):
            """Close the designated port"""
            self.stop()
            self.ser.close()
    
    def clear_errors(self):
            """Reset errors"""
            self.ser.write("CLRFAULT\r\n")
            self.ser.write("OPMODE 0\r\n EN\r\n")
