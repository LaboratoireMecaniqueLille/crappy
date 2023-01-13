# coding: utf-8

from typing import Dict, Type

from .ft232h import Motorkit_pump_ft232h

from .meta_actuator import MetaActuator, Actuator
from .biaxe import Biaxe
from .biotens import Biotens
from .cmDrive import CM_drive
from .fakemotor import Fake_motor
from .motorkit_pump import Motorkit_pump
from .oriental import Oriental
from .servostar import Servostar
from .pololu_tic import Pololu_tic
from .tra6ppd import Tra6ppd

actuator_dict: Dict[str, Type[Actuator]] = MetaActuator.classes
