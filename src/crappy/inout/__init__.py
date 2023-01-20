# coding: utf-8

from typing import Dict, Type

from .meta_inout import InOut, MetaIO

from .ft232h import ADS1115FT232H
from .ft232h import GPIOSwitchFT232H
from .ft232h import MCP9600FT232H
from .ft232h import MPRLSFT232H
from .ft232h import NAU7802FT232H
from .ft232h import WaveshareADDAFT232H

from .ads1115 import ADS1115
from .agilent_34420A import Agilent34420a
from .comedi import Comedi
from .fake_inout import FakeInout
from .gpio_pwm import GPIOPWM
from .gpio_switch import GPIOSwitch
from .gsm import GSM
from .kollmorgen import Koll
from .labjack_t7 import LabjackT7
from .labjack_ue9 import LabjackUE9
from .mcp9600 import MCP9600
from .mprls import MPRLS
from .nau7802 import NAU7802
from .ni_daqmx import NIDAQmx
from .opsens import OpSens
from .pijuice import PiJuice
from .spectrum import Spectrum
from .labjack_t7_streamer import T7Streamer
from .waveshare_ad_da import WaveshareADDA
from .waveshare_high_precision import WaveshareHighPrecision

# Win specific
from .daqmx import DAQmx

# All the inout objects
inout_dict: Dict[str, Type[InOut]] = MetaIO.classes
