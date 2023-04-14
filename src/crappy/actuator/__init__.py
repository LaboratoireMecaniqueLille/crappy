# coding: utf-8

from typing import Dict, Type

from .ft232h import DCMotorHatFT232H

from .meta_actuator import MetaActuator, Actuator
from .jvl_mac_140 import JVLMac140
from .schneider_mdrive_23 import SchneiderMDrive23
from .fake_motor import FakeMotor
from .adafruit_dc_motor_hat import DCMotorHat
from .oriental_ard_k import OrientalARDK
from .kollmorgen_servostar_300 import ServoStar300
from .pololu_tic import PololuTic
from .newport_tra6ppd import NewportTRA6PPD

actuator_dict: Dict[str, Type[Actuator]] = MetaActuator.classes
