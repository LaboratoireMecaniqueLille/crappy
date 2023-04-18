# coding: utf-8

from typing import Dict, Type

# Parent class
from .meta_camera import Camera, MetaCamera, camera_setting
# Virtual cameras
from .fake_camera import FakeCamera
from .file_reader import FileReader
# Physical cameras
from .opencv_basic_camera import Webcam
from .ximea_xiapi import XiAPI
from .raspberry_pi_camera import RaspberryPiCamera
from .gstreamer_camera import CameraGstreamer
from .opencv_camera import CameraOpencv
# Cameralink cameras
from .cameralink import CLCamera
from .cameralink import BiSpectral
from .cameralink import Jai, Jai8
from .seek_thermal_pro import SeekThermalPro

camera_dict: Dict[str, Type[Camera]] = MetaCamera.classes
