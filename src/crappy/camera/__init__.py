# coding: utf-8

from typing import Dict, Type

from .fake_camera import FakeCamera
from .file_reader import FileReader
from .gstreamer_camera import CameraGstreamer
from .opencv_camera import CameraOpencv
from .opencv_camera_basic import Webcam
from .raspberry_pi_camera import RaspberryPiCamera
from .seek_thermal_pro import SeekThermalPro
from .ximea_xiapi import XiAPI

from .cameralink import BaslerIronmanCameraLink
from .cameralink import BiSpectral
from .cameralink import JaiGO5000CPMCL, JaiGO5000CPMCL8Bits

from .meta_camera import Camera, MetaCamera, camera_setting

camera_dict: Dict[str, Type[Camera]] = MetaCamera.classes
