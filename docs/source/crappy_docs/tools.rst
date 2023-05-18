=====
Tools
=====

Microcontroller templates
-------------------------

Arduino Template
++++++++++++++++
The `src/crappy/tool/microcontroller.ino` file is an Arduino template meant
to be used in combination with the :class:`~crappy.blocks.ClientServer` Block.
It greatly simplifies the use of this Block by leaving only a few fields for
the user to complete. It mainly manages the serial communication between the PC
and the microcontroller.

MicroPython Template
++++++++++++++++++++
The `src/crappy/tool/microcontroller.py` file is a MicroPython template meant
to be used in combination with the :class:`~crappy.blocks.ClientServer` Block.
It greatly simplifies the use of this Block by leaving only a few fields for
the user to complete. It mainly manages the serial communication between the PC
and the microcontroller.

Bindings
--------

Comedi Bind
+++++++++++
.. automodule:: crappy.tool.bindings.comedi_bind

Py Spectrum
+++++++++++
.. automodule:: crappy.tool.bindings.pyspcm

Camera Configurators
--------------------

Camera Configurator
+++++++++++++++++++
.. autoclass:: crappy.tool.camera_config.CameraConfig
   :members: main, log
   :special-members: __init__

Camera Configurator with Boxes
++++++++++++++++++++++++++++++
.. autoclass:: crappy.tool.camera_config.CameraConfigBoxes
   :special-members: __init__

DIS Correl Configurator
+++++++++++++++++++++++
.. autoclass:: crappy.tool.camera_config.DISCorrelConfig
   :special-members: __init__

DIS VE Configurator
+++++++++++++++++++
.. autoclass:: crappy.tool.camera_config.DICVEConfig
   :special-members: __init__

Video Extenso Configurator
++++++++++++++++++++++++++
.. autoclass:: crappy.tool.camera_config.VideoExtensoConfig
   :special-members: __init__

Configurator Tools
++++++++++++++++++

Box
"""
.. autoclass:: crappy.tool.camera_config.config_tools.Box
   :members: no_points, reset, sorted
   :special-members: __init__

Histogram Process
"""""""""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.HistogramProcess
   :members: run, log
   :special-members: __init__

Spots Boxes
"""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.SpotsBoxes
   :members: set_spots, empty, reset
   :special-members: __init__

Spots Detector
""""""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.SpotsDetector
   :members: detect_spots
   :special-members: __init__

Zoom
""""
.. autoclass:: crappy.tool.camera_config.config_tools.Zoom
   :members: reset, update_zoom, update_move
   :special-members: __init__

Data
----
The folder `src/crappy/tool/data/` contains various images that need to be
distributed with the module. The `no_image.png` image is used by the
:class:`~crappy.tool.camera_config.CameraConfig` window in case no image could
be acquired yet. The `speckle.png` and `ve_markers.tif` images serve as example
of samples with respectively a speckle and spots drawn on them. They are used
in several examples to demonstrate the use of
:class:`~crappy.blocks.VideoExtenso` or :class:`~crappy.blocks.DICVE` without
requiring any camera. The `pad.png` image is used for demonstrating the
use of the :class:`~crappy.blocks.Canvas` Block.

FT232H Tools
------------

FT232H
++++++
.. autoclass:: crappy.tool.ft232h.FT232H
   :members: write_byte, write_byte_data, write_word_data, write_block_data,
             write_i2c_block_data, read_byte, read_byte_data, read_word_data,
             read_i2c_block_data, i2c_rdwr, bits_per_word, cshigh, loop, no_cs,
             lsbfirst, max_speed_hz, mode, threewire, readbytes, writebytes,
             writebytes2, xfer, xfer2, xfer3, get_gpio, set_gpio, close, log
   :special-members: __init__

FT232H Server
+++++++++++++
.. autoclass:: crappy.tool.ft232h.FT232HServer
   :members: close
   :special-members: __init__

I2C Message
+++++++++++
.. autoclass:: crappy.tool.ft232h.I2CMessage
   :members: read, write, addr, buf, len
   :special-members: __init__

USB Server
++++++++++
.. autoclass:: crappy.tool.ft232h.USBServer
   :members: register, start_server, stop_server, run, log
   :special-members: __init__

Image Processing Tools
----------------------

DIS Correl Tool
+++++++++++++++
.. autoclass:: crappy.tool.image_processing.DISCorrelTool
   :members: set_img0, set_box, get_data
   :special-members: __init__

DIS VE Tool
+++++++++++
.. autoclass:: crappy.tool.image_processing.DICVETool
   :members: set_img0, calculate_displacement
   :special-members: __init__

Fields Tools
++++++++++++
.. autofunction:: crappy.tool.image_processing.fields.get_field
.. autofunction:: crappy.tool.image_processing.fields.get_res

GPU Correl Tool
+++++++++++++++
.. autoclass:: crappy.tool.image_processing.GPUCorrelTool
   :members: set_img_size, set_orig, prepare, get_disp, get_data_display,
             get_res, clean
   :special-members: __init__

GPU Kernels
+++++++++++
The `src/crappy/tool/image_processing/kernels.cu` file contains the default
kernels to use with :mod:`pycuda`. They're used by the
:class:`~crappy.tool.image_processing.GPUCorrelTool` if no other kernel file is
provided.

Video Extenso Tool
++++++++++++++++++
.. autoclass:: crappy.tool.image_processing.video_extenso.VideoExtensoTool
   :members: start_tracking, stop_tracking, get_data
   :special-members: __init__

Video Extenso Tracker
+++++++++++++++++++++
.. autoclass:: crappy.tool.image_processing.video_extenso.tracker.Tracker
   :members: run
   :special-members: __init__
