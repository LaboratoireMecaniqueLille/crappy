# coding: utf-8

from .inout import InOut, MetaIO

from .ads1115 import Ads1115
from .agilent34420A import Agilent34420a
from .comedi import Comedi
from .fake_inout import Fake_inout
from .gpio_pwm import Gpio_pwm
from .gpio_switch import Gpio_switch
from .gsm import Gsm
from .kollmorgen import Koll
from .labjackT7 import Labjack_t7
from .labjackUE9 import Labjack_ue9
from .mcp9600 import Mcp9600
from .mprls import Mprls
from .nau7802 import Nau7802
from .ni_daqmx import Nidaqmx
from .open_daq import Opendaq
from .opsens import Opsens
from .piJuice import Pijuice
from .spectrum import Spectrum
from .t7Streamer import T7_streamer
from .waveshare_ad_da import Waveshare_ad_da
from .waveshare_ad_da_ft232h import Waveshare_ad_da_ft232h
from .waveshare_high_precision import Waveshare_high_precision

# Win specific
from .daqmx import Daqmx

# All the inout objects
inout_dict = MetaIO.classes
