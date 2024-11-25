# coding: utf-8

from .fake_camera import FakeCamera
from .file_reader import FileReader
from .opencv_camera_webcam import Webcam
from .raspberry_pi_camera import RaspberryPiCamera
from .raspberry_pi_camera_2 import RaspberryPiCamera2
from .seek_thermal_pro import SeekThermalPro
from .ximea_xiapi import XiAPI

from .cameralink import BaslerIronmanCameraLink
from .cameralink import JaiGO5000CPMCL, JaiGO5000CPMCL8Bits

from .meta_camera import Camera, MetaCamera, camera_setting

from platform import system
from subprocess import run
if system() == 'Linux':
  try:
    run(['v4l2-ctl'], capture_output=True)
  except FileNotFoundError:
    from .gstreamer_camera_basic import CameraGstreamer
    from .opencv_camera_basic import CameraOpencv
  else:
    from .gstreamer_camera_v4l2 import CameraGstreamer
    from .opencv_camera_v4l2 import CameraOpencv
else:
  from .gstreamer_camera_basic import CameraGstreamer
  from .opencv_camera_basic import CameraOpencv

from ._deprecated import deprecated_cameras
camera_dict: dict[str, type[Camera]] = MetaCamera.classes
