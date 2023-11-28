# coding: utf-8

from typing import Dict, Type

from .adafruit_dc_motor_hat import DCMotorHat
from .fake_dc_motor import FakeDCMotor
from .fake_stepper_motor import FakeStepperMotor
from .jvl_mac_140 import JVLMac140
from .kollmorgen_servostar_300 import ServoStar300
from .newport_tra6ppd import NewportTRA6PPD
from .oriental_ard_k import OrientalARDK
from .pololu_tic import PololuTic
from .schneider_mdrive_23 import SchneiderMDrive23

from .ft232h import DCMotorHatFT232H

from .meta_actuator import MetaActuator, Actuator

from ._deprecated import deprecated_actuators
actuator_dict: Dict[str, Type[Actuator]] = MetaActuator.classes
