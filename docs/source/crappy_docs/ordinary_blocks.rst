===============
Ordinary Blocks
===============

These Blocks exchange labeled data through regular Links. Image-oriented
all-in-one Blocks are included here because they also appear as single nodes
in the experiment graph. For composable image stages, see
:doc:`vision_blocks`.

.. autosummary::
   :nosignatures:

   crappy.blocks.AutoDriveVideoExtenso
   crappy.blocks.Button
   crappy.blocks.Camera
   crappy.blocks.Canvas
   crappy.blocks.ClientServer
   crappy.blocks.Dashboard
   crappy.blocks.DISCorrel
   crappy.blocks.DICVE
   crappy.blocks.FakeMachine
   crappy.blocks.Generator
   crappy.blocks.GPUCorrel
   crappy.blocks.GPUVE
   crappy.blocks.Grapher
   crappy.blocks.HDFRecorder
   crappy.blocks.IOBlock
   crappy.blocks.LinkReader
   crappy.blocks.Machine
   crappy.blocks.MeanBlock
   crappy.blocks.Multiplexer
   crappy.blocks.Pause
   crappy.blocks.PID
   crappy.blocks.Recorder
   crappy.blocks.Sink
   crappy.blocks.StopBlock
   crappy.blocks.StopButton
   crappy.blocks.Synchronizer
   crappy.blocks.UController
   crappy.blocks.VideoExtenso

.. _crappy_docs/blocks:auto drive:

Auto Drive
----------

.. autoclass:: crappy.blocks.AutoDriveVideoExtenso
   :members: __init__, prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:button:

Button
------

.. autoclass:: crappy.blocks.Button
   :members: prepare, begin, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:camera block:

Camera Block
------------

.. autoclass:: crappy.blocks.Camera
   :members: prepare, begin, loop, finish, configure
   :special-members: __init__
   :private-members: _configure

.. _crappy_docs/blocks:canvas:

Canvas
------

.. autoclass:: crappy.blocks.Canvas
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:client server:

Client Server
-------------

.. autoclass:: crappy.blocks.ClientServer
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:dashboard:

Dashboard
---------

.. autoclass:: crappy.blocks.Dashboard
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:dis correl:

DIS Correl
----------

.. autoclass:: crappy.blocks.DISCorrel
   :members: prepare
   :special-members: __init__

.. _crappy_docs/blocks:dic ve:

DIC VE
------

.. autoclass:: crappy.blocks.DICVE
   :members: prepare
   :special-members: __init__

.. _crappy_docs/blocks:fake machine:

Fake Machine
------------

.. autoclass:: crappy.blocks.FakeMachine
   :members: prepare, begin, loop
   :special-members: __init__

.. _crappy_docs/blocks:generator:

Generator
---------

.. autoclass:: crappy.blocks.Generator
   :members: prepare, begin, loop
   :special-members: __init__

.. _crappy_docs/blocks:gpu correl:

GPU Correl
----------

.. autoclass:: crappy.blocks.GPUCorrel
   :members: prepare
   :special-members: __init__

.. _crappy_docs/blocks:gpu ve:

GPU VE
------

.. autoclass:: crappy.blocks.GPUVE
   :members: prepare
   :special-members: __init__

.. _crappy_docs/blocks:grapher:

Grapher
-------

.. autoclass:: crappy.blocks.Grapher
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:hdf recorder:

HDF Recorder
------------

.. autoclass:: crappy.blocks.HDFRecorder
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:ioblock:

IOBlock
-------

.. autoclass:: crappy.blocks.IOBlock
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:link reader:

Link Reader
-----------

.. autoclass:: crappy.blocks.LinkReader
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:machine:

Machine
-------

.. autoclass:: crappy.blocks.Machine
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:mean block:

Mean Block
----------

.. autoclass:: crappy.blocks.MeanBlock
   :members: prepare, begin, loop
   :special-members: __init__

.. _crappy_docs/blocks:multiplexer:

Multiplexer
-----------

.. autoclass:: crappy.blocks.Multiplexer
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:pause block:

Pause Block
-----------

.. autoclass:: crappy.blocks.Pause
   :members: prepare, loop
   :special-members: __init__

.. _crappy_docs/blocks:pid:

PID
---

.. autoclass:: crappy.blocks.PID
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:recorder:

Recorder
--------

.. autoclass:: crappy.blocks.Recorder
   :members: prepare, loop
   :special-members: __init__

.. _crappy_docs/blocks:sink:

Sink
----

.. autoclass:: crappy.blocks.Sink
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:stop block:

Stop Block
----------

.. autoclass:: crappy.blocks.StopBlock
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:stop button:

Stop Button
-----------

.. autoclass:: crappy.blocks.StopButton
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:synchronizer:

Synchronizer
------------

.. autoclass:: crappy.blocks.Synchronizer
   :members: loop
   :special-members: __init__

.. _crappy_docs/blocks:ucontroller:

UController
-----------

.. autoclass:: crappy.blocks.UController
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:video extenso:

Video Extenso
-------------

.. autoclass:: crappy.blocks.VideoExtenso
   :members: prepare
   :special-members: __init__
