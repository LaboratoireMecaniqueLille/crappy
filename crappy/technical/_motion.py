# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup Motion Motion
# @{

## @file _motion.py
# @brief Class to control a motor defined in crappy.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

import serial
import time
from ._meta import motion
from crappy.technical import __motors__ as motors


class Motion(motion.Motion):
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, motor_name="servostar", *args, **kwargs):

        if not motor_name.capitalize() in [x.capitalize() for x in motors]:
            while not motor_name.capitalize() in [x.capitalize() for x in motors]:
                print "Unreconized motor name: %s" % motor_name
                print "motor_name should be one of these: "
                print motors, '\n'
                motor_name = raw_input('motor_name (ENTER to resume): ')
                print '\n'
                if len(motor_name) == 0:
                    print "Unreconized motor name, leaving program..."
                    return
        try:
            module = __import__("crappy.technical", fromlist=["%sTechnical" % motors[0]])
            motor_name = motors[[x.capitalize() for x in motors].index(motor_name.capitalize())]
            self.Motor = getattr(module, "%s" % motor_name)
        except Exception as e:
            print "{0}".format(e), " : Unreconized motor name, leaving program...\n"
            return

        self.port = port
        self.baudrate = baudrate
        self.technical = self.Motor(port=port, baudrate=self.baudrate, *args, **kwargs)
        self.actuator = self.technical.actuator
        self.sensor = self.technical.sensor

    """Stop the motor motion"""

    def stop(self):
        self.technical.stop()

    """Reset the serial communication, before reopen it to set displacement to zero"""

    def reset(self):
        self.technical.reset()

    def close(self):
        self.technical.stop()

    def clear_errors(self):
        self.technical.clear_errors()

    def set_speed(self, speed):
        self.technical.actuator.set_speed(speed)

    def set_position(self, position, speed=None):
        self.technical.actuator.set_position(position, speed)

    def get_position(self):
        self.technical.sensor.get_position()


