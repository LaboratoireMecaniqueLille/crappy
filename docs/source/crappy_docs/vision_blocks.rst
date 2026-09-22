============
VisionBlocks
============

VisionBlocks exchange images through ImageLinks and can also use regular Links
for measurements, commands, and metadata.

.. autosummary::
   :nosignatures:

   crappy.blocks.vision.block.VisionBlock
   crappy.blocks.vision.CameraSource
   crappy.blocks.vision.DICVEProcessor
   crappy.blocks.vision.DISCorrelProcessor
   crappy.blocks.vision.ImageDisplayer
   crappy.blocks.vision.ImageRecorder
   crappy.blocks.vision.VideoExtensoProcessor

.. currentmodule:: crappy.blocks.vision

.. _crappy_docs/blocks:vision block:

VisionBlock
-----------

.. automodule:: crappy.blocks.vision.block

.. currentmodule:: crappy.blocks.vision.block

.. autoclass:: VisionBlock
   :members: prepare, begin, finish, add_img_output, add_img_input, send_img,
             receive_imgs, request_config, add_config_request_in,
             config_requests_in, send_config, add_config_request_out,
             recv_configs, set_shared_objects
   :special-members: __init__

.. py:attribute:: VisionBlock.last_received

   Mapping from each input ImageLink name to its latest locally copied
   :class:`~crappy.blocks.vision.block.ImgData`.

.. _crappy_docs/blocks:camera source:

Camera Source
-------------

.. autoclass:: crappy.blocks.vision.CameraSource
   :members: prepare, loop, finish, configure, default_configuration
   :special-members: __init__

.. _crappy_docs/blocks:dic ve processor:

DIC VE Processor
----------------

.. autoclass:: crappy.blocks.vision.DICVEProcessor
   :members: prepare, loop, request_config
   :special-members: __init__

.. _crappy_docs/blocks:dis correl processor:

DIS Correl Processor
--------------------

.. autoclass:: crappy.blocks.vision.DISCorrelProcessor
   :members: prepare, loop, request_config
   :special-members: __init__

.. _crappy_docs/blocks:image displayer:

Image Displayer
---------------

.. autoclass:: crappy.blocks.vision.ImageDisplayer
   :members: prepare, loop, finish
   :special-members: __init__

.. _crappy_docs/blocks:image recorder:

Image Recorder
--------------

.. autoclass:: crappy.blocks.vision.ImageRecorder
   :members: prepare, loop
   :special-members: __init__

.. _crappy_docs/blocks:video extenso processor:

Video Extenso Processor
-----------------------

.. autoclass:: crappy.blocks.vision.VideoExtensoProcessor
   :members: prepare, loop, finish, request_config
   :special-members: __init__

.. _crappy_docs/blocks:vision block data classes:

Supporting data classes
-----------------------

These classes describe configuration requests and locally received images.
They are documented for authors of custom VisionBlocks.

Configuration Request
+++++++++++++++++++++

.. autoclass:: crappy.blocks.vision.block.ConfigRequest
   :members: requester, args, kwargs, configurator, img_source, connection,
             completed, required
   :undoc-members:

Received Image Data
+++++++++++++++++++

.. autoclass:: crappy.blocks.vision.block.ImgData(id=-1, metadata=None, img=...)
   :members: id, metadata, img
   :undoc-members:
