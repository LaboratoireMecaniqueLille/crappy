#coding: utf-8
from __future__ import print_function

from .._global import NotInstalled

from .camera import Camera, MetaCam
from .fakeCamera import Fake_camera
from .webcam import Webcam
from .ximea import Ximea

try:
  from .ximeaCV import XimeaCV
except ImportError:
  XimeaCV = NotInstalled("XimeaCV")

try:
  from .cameralink import CLCamera
  from .jai import Jai,Jai8
  from .bispectral import Bispectral
except ImportError:
  CLCamera = NotInstalled("CLCamera")
  Jai = NotInstalled("Jai")
  Jai8 = NotInstalled("Jai8")
  Bispectral = NotInstalled("Bispectral")

camera_list = MetaCam.classes
