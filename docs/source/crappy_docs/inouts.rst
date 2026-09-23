========
In / Out
========

In/Out drivers
--------------

Drivers whose documented path starts with ``crappy.collection`` are retained
for compatibility but are not actively maintained. Import
``crappy.collection`` before selecting one of them in an IOBlock.

ADS1115
+++++++
.. autoclass:: crappy.inout.ADS1115
   :members: open, get_data, close
   :special-members: __init__

Agilent 34420A
++++++++++++++
.. autoclass:: crappy.collection.inout.agilent_34420A.Agilent34420a
   :members: open, get_data, close
   :special-members: __init__

Comedi
++++++
.. autoclass:: crappy.collection.inout.comedi.Comedi
   :members: open, set_cmd, get_data, close, make_zero
   :special-members: __init__

DAQmx
+++++
.. autoclass:: crappy.inout.DAQmx
   :members: open, set_cmd, get_data, close, make_zero
   :special-members: __init__

Eurotherm EPC3008
+++++++++++++++++
.. autoclass:: crappy.collection.inout.eurotherm_EPC3008.EurothermEPC3008
   :members: open, get_data, set_cmd, close
   :special-members: __init__

Fake Inout
++++++++++
.. autoclass:: crappy.inout.FakeInOut
   :members: open, set_cmd, get_data, start_stream, get_stream, stop_stream,
             close
   :special-members: __init__

Flow Controller Alicat
++++++++++++++++++++++
.. autoclass:: crappy.collection.inout.flow_controller_alicat.FlowControllerAlicat
   :members: open, get_data, set_cmd, close
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
.. autoclass:: crappy.collection.inout.kollmorgen_akd_pdmm.KollmorgenAKDPDMM
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
.. autoclass:: crappy.collection.inout.labjack_ue9.LabjackUE9
   :members: open, get_data, close, make_zero
   :special-members: __init__

MCP9600
+++++++
.. autoclass:: crappy.collection.inout.mcp9600.MCP9600
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
.. autoclass:: crappy.collection.inout.opsens_handysens.HandySens
   :members: open, get_data, close
   :special-members: __init__

Phidget Wheatstone Bridge
+++++++++++++++++++++++++
.. autoclass:: crappy.inout.PhidgetWheatstoneBridge
   :members: open, get_data, close
   :special-members: __init__

PiJuice
+++++++
.. autoclass:: crappy.collection.inout.pijuice_hat.PiJuice
   :members: open, get_data, close
   :special-members: __init__

Sager SG-GS1700
+++++++++++++++
.. autoclass:: crappy.collection.inout.sager_sg_gs1700.SagerSG_GS1700
   :members: open, get_data, set_cmd, close
   :special-members: __init__

Sim868
++++++
.. autoclass:: crappy.collection.inout.sim868.Sim868
   :members: open, set_cmd, close
   :special-members: __init__

Spectrum M2I 4711
+++++++++++++++++
.. autoclass:: crappy.collection.inout.spectrum_m2i4711.SpectrumM2I4711
   :members: open, start_stream, get_stream, stop_stream, close
   :special-members: __init__

Waveshare AD/DA
+++++++++++++++
.. autoclass:: crappy.collection.inout.waveshare_ad_da.WaveshareADDA
   :members: open, set_cmd, get_data, close
   :special-members: __init__

Waveshare High Precision
++++++++++++++++++++++++
.. autoclass:: crappy.collection.inout.waveshare_high_precision.WaveshareHighPrecision
   :members: open, get_data, close
   :special-members: __init__

Parent In/Out
-------------

InOut
+++++
.. automodule:: crappy.inout.meta_inout.inout

.. currentmodule:: crappy.inout.meta_inout.inout

.. autoclass:: InOut
   :members: open, set_cmd, get_data, start_stream, get_stream, stop_stream,
             close, log, make_zero, return_data, return_stream
   :special-members: __init__
