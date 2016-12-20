#!/usr/bin/python
# -*- coding: utf-8 -*-
##  @addtogroup actuator
# @{

##  @defgroup VariateurTriboActuator VariateurTriboActuator
# @{

## @ _variateurTriboActuator.py
# @brief TODO
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

import time

from ._meta import motion
from .._warnings import deprecated as deprecated


class VariateurTriboActuator(motion.MotionActuator):
  def __init__(self, ser_servostar=None):
    super(VariateurTriboActuator, self).__init__()
    self.ser_servostar = ser_servostar  # [Deprecated]
    self.ser = self.ser_servostar
    self.init = False

  def set_speed(self, speed):
    # TODO
    pass

  @deprecated(None, "Use stop method defined in VariateurTribo instead.")
  def stop_motor(self):
    self.ser.write('dis\r\n')

  def set_mode_position(self):
    self.ser.write('opmode 8\r\n')
    self.mode = 'position'
    time.sleep(0.1)
    # print self.mode

  def set_position(self, position, speed=20000, acc=200, dec=200):
    ## creating the order for the motor example ORDER 0 1000 20000 8192 200 200 0 0 0 0\r\n
    self.ser.write("ORDER 0 " + str(position) + " " +
                   str(speed) + " 8192 " + str(acc) + " " + str(dec) + " " + " 0 0 0 0\r\n")
    self.ser.write("MOVE 0\r\n")  # activates the order

  @deprecated(set_position)
  def go_position(self, position, speed=20000, acc=200, dec=200):
    ##
    # DEPRECATED: Use set_position instead.
    #
    self.set_position(position, speed, acc, dec)

  def set_mode_analog(self):
    self.ser.write('opmode 1\r\n')
    self.ser.write('ancnfg 0\r\n')
    time.sleep(0.1)
    self.mode = 'effort'
    # print self.mode

  def initialisation(self):
    self.ser.write('opmode 8\r\n')
    self.ser.write('en\r\n')
    self.ser.write('mh\r\n')
    self.init = True
