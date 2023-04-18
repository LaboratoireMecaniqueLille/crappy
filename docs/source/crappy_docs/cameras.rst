=======
Cameras
=======

Regular Cameras
---------------

Fake Camera
+++++++++++
.. autoclass:: crappy.camera.FakeCamera
   :members: open, get_image
   :special-members: __init__

File Reader
+++++++++++
.. autoclass:: crappy.camera.FileReader
   :members: open, get_image
   :special-members: __init__

Camera GStreamer
++++++++++++++++
.. autoclass:: crappy.camera.CameraGstreamer
   :members: open, get_image, close
   :special-members: __init__

Camera OpenCV
+++++++++++++
.. autoclass:: crappy.camera.CameraOpencv
   :members: open, get_image, close
   :special-members: __init__

Raspberry Pi Camera
+++++++++++++++++++
.. autoclass:: crappy.camera.RaspberryPiCamera
   :members: open, get_image, close
   :special-members: __init__

Seek Thermal Pro
++++++++++++++++
.. autoclass:: crappy.camera.SeekThermalPro
   :members: open, get_image, close
   :special-members: __init__

Webcam
++++++
.. autoclass:: crappy.camera.Webcam
   :members: open, get_image, close
   :special-members: __init__

Xi API
++++++
.. autoclass:: crappy.camera.XiAPI
   :members: open, get_image, close
   :special-members: __init__

CameraLink Cameras
------------------

Bi Spectral
+++++++++++
.. autoclass:: crappy.camera.cameralink.BiSpectral
   :members: open, get_image
   :special-members: __init__

Camera Link
+++++++++++
.. autoclass:: crappy.camera.cameralink.CLCamera
   :members: open, get_image, close
   :special-members: __init__

JAI
+++
.. autoclass:: crappy.camera.cameralink.Jai
   :members: open, get_image
   :special-members: __init__

JAI 8
+++++
.. autoclass:: crappy.camera.cameralink.Jai8
   :members: open
   :special-members: __init__

Parent Camera
-------------

Camera
++++++
.. autoclass:: crappy.camera.Camera
   :members: open, get_image, close, log, add_bool_setting, add_choice_setting,
             add_scale_setting, add_trigger_setting, add_software_roi,
             reload_software_roi, apply_soft_roi, set_all
   :special-members: __init__, __getattr__, __setattr__

Meta Camera
+++++++++++
.. autoclass:: crappy.camera.MetaCamera
   :special-members: __init__

Camera Settings
+++++++++++++++

Camera Setting
""""""""""""""
.. autoclass:: crappy.camera.meta_camera.camera_setting.CameraSetting
   :members: value, log, reload
   :special-members: __init__

Camera Bool Setting
"""""""""""""""""""
.. autoclass:: crappy.camera.meta_camera.camera_setting.CameraBoolSetting
   :special-members: __init__

Camera Choice Setting
"""""""""""""""""""""
.. autoclass:: crappy.camera.meta_camera.camera_setting.CameraChoiceSetting
   :members: reload
   :special-members: __init__

Camera Scale Setting
""""""""""""""""""""
.. autoclass:: crappy.camera.meta_camera.camera_setting.CameraScaleSetting
   :members: value, reload
   :special-members: __init__
