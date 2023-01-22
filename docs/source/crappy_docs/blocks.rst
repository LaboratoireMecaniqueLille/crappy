======
Blocks
======

Regular Blocks
--------------

Auto Drive
++++++++++
.. autoclass:: crappy.blocks.AutoDrive
   :members: __init__, prepare, loop, finish
   :special-members: __init__

Camera Block
++++++++++++
.. autoclass:: crappy.blocks.Camera
   :members: prepare, begin, loop, finish
   :special-members: __init__

Client Server
+++++++++++++
.. autoclass:: crappy.blocks.ClientServer
   :members: prepare, loop, finish
   :special-members: __init__

Dashboard
+++++++++
.. autoclass:: crappy.blocks.Dashboard
   :members: prepare, loop, finish
   :special-members: __init__

DIS Correl
++++++++++
.. autoclass:: crappy.blocks.DISCorrel
   :members: prepare
   :special-members: __init__

DIS VE
++++++
.. autoclass:: crappy.blocks.DICVE
   :members: prepare
   :special-members: __init__

Drawing
+++++++
.. autoclass:: crappy.blocks.Drawing
   :members: prepare, loop, finish
   :special-members: __init__

Fake Machine
++++++++++++
.. autoclass:: crappy.blocks.FakeMachine
   :members: prepare, begin, loop
   :special-members: __init__

Generator
+++++++++
.. autoclass:: crappy.blocks.Generator
   :members: prepare, begin, loop
   :special-members: __init__

GPU Correl
++++++++++
.. autoclass:: crappy.blocks.GPUCorrel
   :members: prepare
   :special-members: __init__

GPU VE
++++++
.. autoclass:: crappy.blocks.GPUVE
   :members: prepare
   :special-members: __init__

Grapher
+++++++
.. autoclass:: crappy.blocks.Grapher
   :members: prepare, loop, finish
   :special-members: __init__

GUI
+++
.. autoclass:: crappy.blocks.GUI
   :members: prepare, begin, loop, finish
   :special-members: __init__

HDF Recorder
++++++++++++
.. autoclass:: crappy.blocks.HDFRecorder
   :members: prepare, loop, finish
   :special-members: __init__

IOBlock
+++++++
.. autoclass:: crappy.blocks.IOBlock
   :members: prepare, loop, finish
   :special-members: __init__

Machine
+++++++
.. autoclass:: crappy.blocks.Machine
   :members: prepare, loop, finish
   :special-members: __init__

Mean Block
++++++++++
.. autoclass:: crappy.blocks.MeanBlock
   :members: prepare, begin, loop
   :special-members: __init__

Multiplexer
+++++++++++
.. autoclass:: crappy.blocks.Multiplex
   :members: loop
   :special-members: __init__

PID
+++
.. autoclass:: crappy.blocks.PID
   :members: loop
   :special-members: __init__

Reader
++++++
.. autoclass:: crappy.blocks.Reader
   :members: loop
   :special-members: __init__

Recorder
++++++++
.. autoclass:: crappy.blocks.Recorder
   :members: prepare, loop
   :special-members: __init__

Sink
++++
.. autoclass:: crappy.blocks.Sink
   :members: loop
   :special-members: __init__

UController
+++++++++++
.. autoclass:: crappy.blocks.UController
   :members: prepare, loop, finish
   :special-members: __init__

Video Extenso
+++++++++++++
.. autoclass:: crappy.blocks.VideoExtenso
   :members: prepare
   :special-members: __init__

Generator Paths
---------------
There are several types of path available for the generator block.

Path
++++
.. autoclass:: crappy.blocks.generator_path.Path
   :members: get_cmd, parse_condition, log
   :special-members: __init__

Constant
++++++++
.. autoclass:: crappy.blocks.generator_path.Constant
   :members: get_cmd
   :special-members: __init__

Custom
++++++
.. autoclass:: crappy.blocks.generator_path.Custom
   :members: get_cmd
   :special-members: __init__

Cyclic
++++++
.. autoclass:: crappy.blocks.generator_path.Cyclic
   :members: get_cmd
   :special-members: __init__

Cyclic Ramp
+++++++++++
.. autoclass:: crappy.blocks.generator_path.CyclicRamp
   :members: get_cmd
   :special-members: __init__

Integrator
++++++++++
.. autoclass:: crappy.blocks.generator_path.Integrator
   :members: get_cmd
   :special-members: __init__

Conditional
+++++++++++
.. autoclass:: crappy.blocks.generator_path.Conditional
   :members: get_cmd
   :special-members: __init__

Ramp
++++
.. autoclass:: crappy.blocks.generator_path.Ramp
   :members: get_cmd
   :special-members: __init__

Sine
++++
.. autoclass:: crappy.blocks.generator_path.Sine
   :members: get_cmd
   :special-members: __init__

Camera Processes
----------------

Camera Process
++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.CameraProcess
   :members: set_shared, run
   :special-members: __init__

DIS Correl Process
++++++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.DISCorrelProcess
   :members: set_box
   :special-members: __init__

DIS VE Process
++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.DICVEProcess
   :special-members: __init__

Displayer Process
+++++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.Displayer
   :special-members: __init__, __del__

GPU Correl Process
++++++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.GPUCorrelProcess
   :special-members: __init__

GPU VE Process
++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.GPUVEProcess
   :special-members: __init__

Recorder Process
++++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.ImageSaver
   :special-members: __init__

Video Extenso Process
+++++++++++++++++++++
.. autoclass:: crappy.blocks.camera_processes.VideoExtensoProcess
   :special-members: __init__

Parent Block
------------

Block
+++++
.. autoclass:: crappy.blocks.Block
   :members: get_name, start_all, prepare_all, renice_all, launch_all,
             stop_all, reset, run, main, prepare, begin, loop, finish, debug,
             t0, add_output, add_input, log, send, data_available, recv_data,
             recv_last_data, recv_all_data, recv_all_data_raw
   :special-members: __init__

Meta Block
++++++++++
.. autoclass:: crappy.blocks.meta_block.MetaBlock
   :special-members: __init__
