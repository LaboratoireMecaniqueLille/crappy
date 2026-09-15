# coding: utf-8

from .fake_dc_motor import FakeDCMotor
from .fake_stepper_motor import FakeStepperMotor
from .jvl_mac_140 import JVLMac140
from .kollmorgen_servostar_300 import ServoStar300
from .phidgets_stepper4a import Phidget4AStepper
from .pololu_tic import PololuTic

from .meta_actuator import Actuator

from ._deprecated import deprecated_actuators
from ._collection import moved_to_collection
actuator_dict: dict[str, type[Actuator]] = Actuator.classes
