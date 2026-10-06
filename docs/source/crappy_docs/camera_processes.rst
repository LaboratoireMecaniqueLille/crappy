====================================
All-in-one Camera processing helpers
====================================

:class:`~crappy.blocks.camera_processes.CameraProcess` is the supported base
for advanced processing inside an all-in-one Camera Block. The concrete
classes below support Crappy's built-in all-in-one Blocks and are primarily
useful when maintaining those integrations.

.. autosummary::
   :nosignatures:

   crappy.blocks.camera_processes.CameraProcess
   crappy.blocks.camera_processes.DICVEProcess
   crappy.blocks.camera_processes.DISCorrelProcess
   crappy.blocks.camera_processes.Displayer
   crappy.blocks.camera_processes.GPUCorrelProcess
   crappy.blocks.camera_processes.GPUVEProcess
   crappy.blocks.camera_processes.ImageSaver
   crappy.blocks.camera_processes.VideoExtensoProcess

.. _crappy_docs/blocks:camera process:

CameraProcess extension base
----------------------------

.. autoclass:: crappy.blocks.camera_processes.CameraProcess
   :members: set_shared, run, init, loop, finish, send, send_to_draw,
             set_config, log
   :special-members: __init__
   :private-members: _get_data

.. _crappy_docs/blocks:dic ve process:

DIC VE Process
--------------

.. autoclass:: crappy.blocks.camera_processes.DICVEProcess
   :members: init, loop, set_config
   :special-members: __init__

.. _crappy_docs/blocks:dis correl process:

DIS Correl Process
------------------

.. autoclass:: crappy.blocks.camera_processes.DISCorrelProcess
   :members: init, loop, set_config
   :special-members: __init__

.. _crappy_docs/blocks:displayer process:

Displayer Process
-----------------

.. autoclass:: crappy.blocks.camera_processes.Displayer
   :members: init, loop, finish
   :special-members: __init__, __del__

.. _crappy_docs/blocks:gpu correl process:

GPU Correl Process
------------------

.. autoclass:: crappy.blocks.camera_processes.GPUCorrelProcess
   :members: init, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:gpu ve process:

GPU VE Process
--------------

.. autoclass:: crappy.blocks.camera_processes.GPUVEProcess
   :members: init, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:recorder process:

Recorder Process
----------------

.. autoclass:: crappy.blocks.camera_processes.ImageSaver
   :members: init, loop
   :special-members: __init__

.. _crappy_docs/blocks:video extenso process:

Video Extenso Process
---------------------

.. autoclass:: crappy.blocks.camera_processes.VideoExtensoProcess
   :members: init, loop, finish, set_config
   :special-members: __init__
