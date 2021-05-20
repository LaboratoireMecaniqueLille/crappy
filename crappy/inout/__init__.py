# coding: utf-8

from .inout import InOut, MetaIO

from .agilent34420A import Agilent34420a
from .arduino import Arduino
from .comedi import Comedi
from .kollmorgen import Koll
from .labjackT7 import Labjack_t7
from .labjackUE9 import Labjack_ue9
from .nidaqmx import Nidaqmx
from .opendaq import Opendaq
from .opsens import Opsens
from .spectrum import Spectrum
from .t7Streamer import T7_streamer
from .mcp9600 import Mcp9600
from .ads1115 import Ads1115
from .nau7802 import Nau7802
from .gpio_switch import Gpio_switch
from .gpio_pwm import Gpio_pwm
from .waveshare_ad_da import Waveshare_ad_da

# Win specific
from .daqmx import Daqmx

inout_list = MetaIO.IOclasses
in_list = MetaIO.Iclasses
in_list.update(inout_list)
out_list = MetaIO.Oclasses
out_list.update(inout_list)
