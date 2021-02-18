#coding: utf-8

# Parent class
from .camera import Camera, MetaCam
# Virtual cameras
from .fakeCamera import Fake_camera
from .streamer import Streamer
# Physical cameras
from .webcam import Webcam
from .xiapi import Xiapi
from .ximeaCV import XimeaCV
# Cameralink cameras
from .cameralink import CLCamera
from .bispectral import Bispectral
from .jai import Jai,Jai8

camera_list = MetaCam.classes
