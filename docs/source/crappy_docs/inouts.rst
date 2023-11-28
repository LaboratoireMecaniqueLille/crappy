========
In / Out
========

Regular In/Outs
---------------

ADS1115
+++++++
.. autoclass:: crappy.inout.ADS1115
   :members: open, get_data, close
   :special-members: __init__

Agilent 34420A
++++++++++++++
.. autoclass:: crappy.inout.Agilent34420a
   :members: open, get_data, close
   :special-members: __init__

Comedi
++++++
.. autoclass:: crappy.inout.Comedi
   :members: open, set_cmd, get_data, close, make_zero
   :special-members: __init__

DAQmx
+++++
.. autoclass:: crappy.inout.DAQmx
   :members: open, set_cmd, get_data, close, make_zero
   :special-members: __init__

Fake Inout
++++++++++
.. autoclass:: crappy.inout.FakeInOut
   :members: open, set_cmd, get_data, start_stream, get_stream, stop_stream,
             close
   :special-members: __init__

GPIO PWM
++++++++
.. autoclass:: crappy.inout.GPIOPWM
   :members: open, set_cmd, close
   :special-members: __init__

GPIO Switch
+++++++++++
.. autoclass:: crappy.inout.GPIOSwitch
   :members: open, set_cmd, close
   :special-members: __init__

Kollmorgen AKD PDMM
+++++++++++++++++++
.. autoclass:: crappy.inout.KollmorgenAKDPDMM
   :members: open, set_cmd, get_data, close
   :special-members: __init__

Labjack T7
++++++++++
.. autoclass:: crappy.inout.LabjackT7
   :members: open, set_cmd, get_data, close, make_zero
   :special-members: __init__

Labjack T7 Streamer
+++++++++++++++++++
.. autoclass:: crappy.inout.T7Streamer
   :members: open, get_data, start_stream, get_stream, stop_stream, close,
             make_zero
   :special-members: __init__

Labjack UE9
+++++++++++
.. autoclass:: crappy.inout.LabjackUE9
   :members: open, get_data, close, make_zero
   :special-members: __init__

MCP9600
+++++++
.. autoclass:: crappy.inout.MCP9600
   :members: open, get_data, close
   :special-members: __init__

MPRLS
+++++
.. autoclass:: crappy.inout.MPRLS
   :members: open, get_data, close
   :special-members: __init__

NAU7802
+++++++
.. autoclass:: crappy.inout.NAU7802
   :members: open, get_data, close
   :special-members: __init__

NI DAQmx
++++++++
.. autoclass:: crappy.inout.NIDAQmx
   :members: open, set_cmd, get_data, start_stream, get_stream, stop_stream,
             close
   :special-members: __init__

OpSens HandySens
++++++++++++++++
.. autoclass:: crappy.inout.HandySens
   :members: open, get_data, close
   :special-members: __init__

Phidget Wheatstone Bridge
+++++++++++++++++++++++++
.. autoclass:: crappy.inout.PhidgetWheatstoneBridge
   :members: open, get_data, close
   :special-members: __init__

PiJuice
+++++++
.. autoclass:: crappy.inout.PiJuice
   :members: open, get_data, close
   :special-members: __init__

Sim868
++++++
.. autoclass:: crappy.inout.Sim868
   :members: open, set_cmd, close
   :special-members: __init__

Spectrum M2I 4711
+++++++++++++++++
.. autoclass:: crappy.inout.SpectrumM2I4711
   :members: open, start_stream, get_stream, stop_stream, close
   :special-members: __init__

Waveshare AD/DA
+++++++++++++++
.. autoclass:: crappy.inout.WaveshareADDA
   :members: open, set_cmd, get_data, close
   :special-members: __init__

Waveshare High Precision
++++++++++++++++++++++++
.. autoclass:: crappy.inout.WaveshareHighPrecision
   :members: open, get_data, close
   :special-members: __init__

FT232H In/Outs
--------------

ADS1115 FT232H
++++++++++++++
.. autoclass:: crappy.inout.ADS1115FT232H
   :members: open, get_data, close
   :special-members: __init__

GPIO Switch FT232H
++++++++++++++++++
.. autoclass:: crappy.inout.GPIOSwitchFT232H
   :members: open, set_cmd, close
   :special-members: __init__

MCP9600 FT232H
++++++++++++++
.. autoclass:: crappy.inout.MCP9600FT232H
   :members: open, get_data, close
   :special-members: __init__

MPRLS FT232H
++++++++++++
.. autoclass:: crappy.inout.MPRLSFT232H
   :members: open, get_data, close
   :special-members: __init__

NAU7802 FT232H
++++++++++++++
.. autoclass:: crappy.inout.NAU7802FT232H
   :members: open, get_data, close
   :special-members: __init__

Waveshare AD/DA FT232H
++++++++++++++++++++++
.. autoclass:: crappy.inout.WaveshareADDAFT232H
   :members: open, set_cmd, get_data, close
   :special-members: __init__

Parent In/Out
-------------

InOut
+++++
.. autoclass:: crappy.inout.InOut
   :members: open, set_cmd, get_data, start_stream, get_stream, stop_stream,
             close, log, make_zero, return_data, return_stream
   :special-members: __init__

Meta InOut
++++++++++
.. autoclass:: crappy.inout.MetaIO
   :special-members: __init__
