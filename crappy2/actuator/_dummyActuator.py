#!/usr/bin/python
# -*- coding: utf-8 -*-
import serial

from crappy2.actuator import motion


class DummyActuator(motion.MotionActuator):
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, num_device=1, conversion_factor=1, ser=None):
        super(DummyActuator, self).__init__()
        self.ser = ser
        self.baudrate = baudrate
        self.num_device = num_device
        self.port = port
        self.conversion_factor = conversion_factor
        self.position = 0.0

    def new(self):
        pass

    def set_speed(self, speed):
        print "speed: ", speed

    def set_home(self):
        print "new home position defined."

    def move_home(self):
        print "moving home."

    def set_position(self, position, speed):
        print "Going to position {0} with speed: {1}".format(position, speed)
        self.position = float(position)

    def get_position(self):
        return self.position