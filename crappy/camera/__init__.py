#coding: utf-8


from .._global import NotInstalled

from .camera import Camera, MetaCam
from .fakeCamera import Fake_camera
from .webcam import Webcam
try:
  from .streamer import Streamer
except ImportError:
  Streamer = NotInstalled("Streamer")
try:
  from .ximea import Ximea
except ImportError:
  Ximea = NotInstalled("Ximea")

try:
  from .ximeaCV import XimeaCV
except (ImportError,AttributeError):
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
