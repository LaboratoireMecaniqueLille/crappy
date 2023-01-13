# coding: utf-8

from typing import Dict, Type

# Parent class
from .camera import Camera, MetaCam
# Virtual cameras
from .fakeCamera import Fake_camera
from .file_reader import File_reader
# Physical cameras
from .webcam import Webcam
from .xiapi import Xiapi
from .pi_camera import Picamera
from .gstreamer import Camera_gstreamer
from .opencv import Camera_opencv
# Cameralink cameras
from .cameralink import Cl_camera
from .bispectral import Bispectral
from .jai import Jai, Jai8
from .seek_thermal_pro import Seek_thermal_pro

camera_dict: Dict[str, Type[Camera]] = MetaCam.classes
