# coding: utf-8

# Parent class
from .camera import Camera, MetaCam
# Virtual cameras
from .fakeCamera import Fake_camera
from .streamer import Streamer
# Physical cameras
from .webcam import Webcam
from .xiapi import Xiapi
from .ximeaCV import XimeaCV
from .picamera_picamera import Picamera
from .picamera_webcam import Picamera_webcam
# Cameralink cameras
from .cameralink import CLCamera
from .bispectral import Bispectral
from .jai import Jai,Jai8
from .seek_thermal_pro import Seek_thermal_pro

camera_list = MetaCam.classes
