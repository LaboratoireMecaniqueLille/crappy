# coding: utf-8

from typing import Dict, Type

# Parent class
from .meta_camera import Camera, MetaCamera, camera_setting
# Virtual cameras
from .fake_camera import FakeCamera
from .file_reader import FileReader
# Physical cameras
from .webcam import Webcam
from .xiapi import XiAPI
from .pi_camera import PiCamera
from .gstreamer import CameraGstreamer
from .opencv import CameraOpencv
# Cameralink cameras
from .cameralink import CLCamera
from .cameralink import BiSpectral
from .cameralink import Jai, Jai8
from .seek_thermal_pro import SeekThermalPro

camera_dict: Dict[str, Type[Camera]] = MetaCamera.classes
