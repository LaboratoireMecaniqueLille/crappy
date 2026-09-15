# coding: utf-8

from .actuator.manifest import DRIVERS as ACTUATOR_DRIVERS
from .camera.manifest import DRIVERS as CAMERA_DRIVERS
from .inout.manifest import DRIVERS as INOUT_DRIVERS

DRIVERS = ACTUATOR_DRIVERS + CAMERA_DRIVERS + INOUT_DRIVERS
