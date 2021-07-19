# coding: utf-8

from .inout import InOut, MetaIO

from .ads1115 import Ads1115
from .agilent34420A import Agilent34420a
from .arduino import Arduino
from .comedi import Comedi
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

# Win specific
from .daqmx import Daqmx

# All the inout objects (either in, out or in and out)
inout_dict = MetaIO.classes

# Only the in AND out classes
inandout_dict = MetaIO.IOclasses
# Only the in classes
in_dict = MetaIO.Iclasses
# Updating it to have all the classes that can take an input
in_dict.update(inandout_dict)
# And same for the out classes
out_dict = MetaIO.Oclasses
out_dict.update(inandout_dict)
