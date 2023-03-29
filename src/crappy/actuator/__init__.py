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
from .kollmorgen_servostar_300 import ServoStar300
from .pololu_tic import PololuTic
from .newport_tra6ppd import NewportTRA6PPD

actuator_dict: Dict[str, Type[Actuator]] = MetaActuator.classes
