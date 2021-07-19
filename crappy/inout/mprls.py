# coding: utf-8
import time
from .inout import InOut
from .._global import OptionalModule

try:
 import adafruit_mprls
except (ImportError, ModuleNotFoundError):
 adafruit_mprls = OptionalModule('adafruit_mprls')

try:
 import board
except (ImportError, ModuleNotFoundError):
 board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')


class Mprls(InOut):
 def __init__(self) -> None:
     InOut.__init__(self)

 def open(self) -> None:
     i2c = board.I2C()
     # Simplest use, connect to default over I2C
     self.mpr = adafruit_mprls.MPRLS(i2c, psi_min=0, psi_max=25)

 def get_data(self) -> list:
     out = [time.time()]
     hpa = self.mpr.pressure
     out.append(hpa)
     return out

 def stop(self) -> None:
     pass

 def close(self) -> None:
     pass
