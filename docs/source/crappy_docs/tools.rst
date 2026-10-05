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

.. automodule:: crappy.tool.camera_config

.. list-table:: Configuration class families
   :header-rows: 1
   :widths: 25 37 38

   * - Shared base
     - Tkinter window
     - PyQt6 window
   * - :class:`~crappy.tool.camera_config.base.camera_config.CameraConfig`
     - :class:`~crappy.tool.camera_config.tkinter.camera_config.TkinterCameraConfig`
     - :class:`~crappy.tool.camera_config.pyqt.camera_config.PyQtCameraConfig`
   * - :class:`~crappy.tool.camera_config.base.camera_config_boxes.CameraConfigBoxes`
     - :class:`~crappy.tool.camera_config.tkinter.camera_config_boxes.TkinterCameraConfigBoxes`
     - :class:`~crappy.tool.camera_config.pyqt.camera_config_boxes.PyQtCameraConfigBoxes`
   * - :class:`~crappy.tool.camera_config.base.dic_ve_config.DICVEConfig`
     - :class:`~crappy.tool.camera_config.tkinter.dic_ve_config.TkinterDICVEConfig`
     - :class:`~crappy.tool.camera_config.pyqt.dic_ve_config.PyQtDICVEConfig`
   * - :class:`~crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig`
     - :class:`~crappy.tool.camera_config.tkinter.dis_correl_config.TkinterDISCorrelConfig`
     - :class:`~crappy.tool.camera_config.pyqt.dis_correl_config.PyQtDISCorrelConfig`
   * - :class:`~crappy.tool.camera_config.base.video_extenso_config.VideoExtensoConfig`
     - :class:`~crappy.tool.camera_config.tkinter.video_extenso_config.TkinterVideoExtensoConfig`
     - :class:`~crappy.tool.camera_config.pyqt.video_extenso_config.PyQtVideoExtensoConfig`

Base configurations
+++++++++++++++++++

.. automodule:: crappy.tool.camera_config.base

.. autoclass:: crappy.tool.camera_config.base.camera_config.CameraConfig
   :members: run, stop, watch_shutdown, get_config, log,
             _create_local_settings, _extra_actions, _validate_close,
             _on_valid_close
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.base.camera_config.ConfigAction
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.base.camera_config_boxes.CameraConfigBoxes
   :members: _on_selection_start, _on_selection_drag, _on_selection_complete,
             _on_selection_end, _handle_box_outside_img
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig
   :members: box, get_config

.. autoclass:: crappy.tool.camera_config.base.dic_ve_config.DICVEConfig
   :members: get_config

.. autoclass:: crappy.tool.camera_config.base.video_extenso_config.VideoExtensoConfig
   :members: get_config

Tkinter configurations
++++++++++++++++++++++

.. automodule:: crappy.tool.camera_config.tkinter

.. autoclass:: crappy.tool.camera_config.tkinter.camera_config.TkinterCameraConfig
   :members: run, start, watch_shutdown, finish, stop
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.tkinter.camera_config_boxes.TkinterCameraConfigBoxes
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.tkinter.dic_ve_config.TkinterDICVEConfig
   :members: get_config
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.tkinter.dis_correl_config.TkinterDISCorrelConfig
   :members: get_config
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.tkinter.video_extenso_config.TkinterVideoExtensoConfig
   :members: get_config
   :special-members: __init__

PyQt6 configurations
++++++++++++++++++++

.. automodule:: crappy.tool.camera_config.pyqt

.. autoclass:: crappy.tool.camera_config.pyqt.camera_config.PyQtCameraConfig
   :members: run, start, watch_shutdown, finish, stop
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.pyqt.camera_config_boxes.PyQtCameraConfigBoxes
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.pyqt.dic_ve_config.PyQtDICVEConfig
   :members: get_config
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.pyqt.dis_correl_config.PyQtDISCorrelConfig
   :members: get_config
   :special-members: __init__

.. autoclass:: crappy.tool.camera_config.pyqt.video_extenso_config.PyQtVideoExtensoConfig
   :members: get_config
   :special-members: __init__

Configurator selection
++++++++++++++++++++++

.. autofunction:: crappy.tool.camera_config.factory.create_configurator

Configurator Tools
++++++++++++++++++

.. automodule:: crappy.tool.camera_config.config_tools

Box
"""
.. autoclass:: crappy.tool.camera_config.config_tools.Box
   :members: no_points, reset, sorted, update, draw
   :special-members: __init__, __post_init__, __add__

Histogram Process
"""""""""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.HistogramProcess
   :members: run, log
   :special-members: __init__

Overlay
"""""""
.. autoclass:: crappy.tool.camera_config.config_tools.Overlay
   :members: draw, log
   :special-members: __init__

Spots Boxes
"""""""""""
.. autoclass:: crappy.tool.camera_config.config_tools.SpotsBoxes
   :members: set_spots, save_length, empty, reset, copy
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
:class:`~crappy.tool.camera_config.base.camera_config.CameraConfig` window in
case no image could be acquired yet. The `speckle.png` and `ve_markers.tif`
images serve as example of samples with respectively a speckle and spots drawn
on them. They are used in several examples to demonstrate the use of
:class:`~crappy.blocks.VideoExtenso` or :class:`~crappy.blocks.DICVE` without
requiring any camera. The `pad.png` image is used for demonstrating the
use of the :class:`~crappy.blocks.Canvas` Block.

Image Processing Tools
----------------------

.. automodule:: crappy.tool.image_processing

Synthetic Strain Image
++++++++++++++++++++++

.. automodule:: crappy.tool.apply_strain_image

.. currentmodule:: crappy.tool.apply_strain_image

.. autoclass:: ApplyStrainToImage
   :special-members: __init__, __call__

This public helper deforms a reference image according to horizontal and
vertical strain values. It is primarily used as the ``image_generator`` of a
:class:`~crappy.blocks.vision.CameraSource` or all-in-one Camera Block in
hardware-free examples. It requires OpenCV.

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

.. _gpu-kernels:

GPU Kernels
+++++++++++
The `src/crappy/tool/image_processing/kernels.cu` file contains the default
kernels to use with :external+pycuda:mod:`pycuda`. They're used by the
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
