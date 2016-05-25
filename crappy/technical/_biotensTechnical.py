# coding: utf-8
import serial
import time
from ..sensor import BiotensSensor
from ..actuator import BiotensActuator
from ._meta import motion

class Biotens(motion.Motion):
    def __init__(self, port='/dev/ttyUSB0', baudrate=19200, size=30):
        """
        Open the connection, and initialise the Biotens.

        You should always use this Class to communicate with the Biotens.

        Parameters
        ----------
        port : str, default = '/dev/ttyUSB0'
            Path to the correct serial port.
        size : int of float, default = 30
            Initial size of your test sample, in mm.
        """
        self.size=size-7
        if self.size<0:
                self.size=0
        self.port=port
        self.baudrate = baudrate
        self.ser=serial.Serial(self.port, baudrate=19200, timeout=0.1)
        self.initialisation()
        self.mise_position()
        self.actuator = BiotensActuator(self.ser)
        self.sensor = BiotensSensor(self.ser)

    def mise_position(self):
        """Set motor into position for sample's placement"""
        self.set_position(self.size+0.3,70) # add 0.2 to ensure going to the wanted position
        startposition='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(10,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(10,'B')+ '\xAA\xAA'
        self.ser.write(startposition)	
        try:
                self.ser.readlines()
        except serial.SerialException:
                pass
        last_position_SI=0
        position_SI=99
        while position_SI!=last_position_SI:
                last_position_SI=position_SI
                position_SI=self.get_position()
                print "position : ", position_SI
        print "Fin"
        self.stop()	
        
        
    def initialisation(self):
        """Actuators goes out completely, in order to set the initial position"""
        
        initposition= '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(38,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(38,'B')+ '\xAA\xAA'
        initspeed = '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(40,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(-50,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(40,'B')+ '\xAA\xAA'
        inittorque = '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(41,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(1023,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(41,'B')+ '\xAA\xAA'	
        toinit= '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(37,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(0,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(37,'B')+ '\xAA\xAA'
        
        self.ser.writelines([initposition, initspeed, inittorque, toinit])
        self.ser.write('\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(12,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(2,'B')+ '\xAA\xAA')
        last_position_SI=0
        position_SI=99
        time.sleep(1)
        while position_SI!=last_position_SI:
                last_position_SI=position_SI
                position_SI=self.get_position()
                print "position : ", position_SI
        print "init done"
        self.stop()	
        #time.sleep(1)
        ### initializes the count when the motors is out.
        startposition='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(10,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(10,'B')+ '\xAA\xAA'
        self.ser.write(startposition)
        #time.sleep(1)
        try:
                self.ser.readlines()
        except serial.SerialException:
                pass
    
    def reset(self):
        #TODO
        pass
    
    def stop(self): 
        """Stop the motor. Amazing."""
        command='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(2,'B')+convert_to_byte(0,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(2,'B')+ '\xAA\xAA'
        self.ser.write(command)
        #return command
        
    def close(self):
        self.stop()
        self.ser.close()
        
    def clear_errors(self): 
        """Clears error in motor registers. obviously."""
        command='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(35,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(35,'B')+ '\xAA\xAA'
        self.ser.write(command)