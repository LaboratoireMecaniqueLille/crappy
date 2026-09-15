# coding: utf-8

from .ads1115 import ADS1115
from .daqmx import DAQmx
from .fake_inout import FakeInOut
from .gpio_pwm import GPIOPWM
from .gpio_switch import GPIOSwitch
from .labjack_t7 import LabjackT7
from .labjack_t7_streamer import T7Streamer
from .mprls import MPRLS
from .nau7802 import NAU7802
from .ni_daqmx import NIDAQmx
from .phidgets_wheatstone_bridge import PhidgetWheatstoneBridge

from .meta_inout import InOut

# All the inout objects
from ._deprecated import deprecated_inouts
from ._collection import moved_to_collection
inout_dict: dict[str, type[InOut]] = InOut.classes
