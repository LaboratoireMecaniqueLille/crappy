# coding: utf-8

from typing import Dict, Type

from .fake_camera import FakeCamera
from .file_reader import FileReader
from .opencv_camera import CameraOpencv
from .opencv_camera_basic import Webcam
from .raspberry_pi_camera import RaspberryPiCamera
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
  else:
    from .gstreamer_camera_v4l2 import CameraGstreamer
else:
  from .gstreamer_camera_basic import CameraGstreamer

camera_dict: Dict[str, Type[Camera]] = MetaCamera.classes
