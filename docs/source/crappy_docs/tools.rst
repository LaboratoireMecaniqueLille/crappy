=====
Tools
=====

Microcontroller templates
-------------------------

Arduino Template
++++++++++++++++
Bla Bla Bla

MicroPython Template
++++++++++++++++++++
Bla Bla Bla

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
.. autoclass:: crappy.tool.camera_config.DISVEConfig
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
   :members: no_points, reset, get_patch, sorted
   :special-members: __init__

Spots Boxes
"""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.SpotsBoxes
   :members: set_spots, empty, reset
   :special-members: __init__

Spots Detector
""""""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.SpotsDetector
   :members: detect_spots, save_length
   :special-members: __init__

Zoom
""""
.. autoclass:: crappy.tool.camera_config.config_tools.Zoom
   :members: reset, update_zoom, update_move
   :special-members: __init__

Data
----
Lorem ipsum

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
.. autoclass:: crappy.tool.image_processing.DISVETool
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
Lorem ipsum

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
