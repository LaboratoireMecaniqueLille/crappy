# coding: utf-8

from .ads1115 import ADS1115
from .agilent_34420A import Agilent34420a
from .comedi import Comedi
from .daqmx import DAQmx
from .fake_inout import FakeInOut
from .gpio_pwm import GPIOPWM
from .gpio_switch import GPIOSwitch
from .kollmorgen_akd_pdmm import KollmorgenAKDPDMM
from .labjack_t7 import LabjackT7
from .labjack_t7_streamer import T7Streamer
from .labjack_ue9 import LabjackUE9
from .mcp9600 import MCP9600
from .mprls import MPRLS
from .nau7802 import NAU7802
from .ni_daqmx import NIDAQmx
from .opsens_handysens import HandySens
from .pijuice_hat import PiJuice
from .phidgets_wheatstone_bridge import PhidgetWheatstoneBridge
from .sim868 import Sim868
from .spectrum_m2i4711 import SpectrumM2I4711
from .waveshare_ad_da import WaveshareADDA
from .waveshare_high_precision import WaveshareHighPrecision

from .ft232h import ADS1115FT232H
from .ft232h import GPIOSwitchFT232H
from .ft232h import MCP9600FT232H
from .ft232h import MPRLSFT232H
from .ft232h import NAU7802FT232H
from .ft232h import WaveshareADDAFT232H

from .meta_inout import InOut, MetaIO

# All the inout objects
from ._deprecated import deprecated_inouts
inout_dict: dict[str, type[InOut]] = MetaIO.classes
