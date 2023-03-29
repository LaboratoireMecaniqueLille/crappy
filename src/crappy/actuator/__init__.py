# coding: utf-8

from typing import Dict, Type

from .ft232h import MotorKitPumpFT232H

from .meta_actuator import MetaActuator, Actuator
from .biaxe import Biaxe
from .biotens import Biotens
from .cm_drive import CMDrive
from .fake_motor import FakeMotor
from .motor_kit_pump import MotorKitPump
from .oriental_ard_k import OrientalARDK
from .servostar import ServoStar
from .pololu_tic import PololuTic
from .tra6ppd import TRA6PPD

actuator_dict: Dict[str, Type[Actuator]] = MetaActuator.classes
