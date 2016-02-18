# coding: utf-8
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from os import path
try: # for autodocumentation
	here = path.abspath(path.dirname(__file__))
	build_path = path.join(here, '../sources/Jai-lib/cllib.so')
	jai = ctypes.CDLL(build_path)
except OSError:
	pass

from ._meta import cameraSensor

class Jai(cameraSensor.CameraSensor):
  
    def __init__(self,numdevice=0, framespersec=99):
        
        self.FPS = framespersec
        self.framespersec=ctypes.c_double(self.FPS)
        self.numdevice = numdevice
        self.configFile = ctypes.c_char_p("../sources/Jai-lib/config.mcf")
        self._init = True
        
    def new(self, exposure=8000, width=2560, height=2048, xoffset=0, yoffset=0, gain=None):
        """
        This method create a new instance of the camera jai class with the class attributes (device number, exposure, width, height, x offset, y offset and FPS)
        Then it prints these settings and initialises the camera and load a configuration file
        """
        self.cam = jai.Camera_new(self.numdevice, self.framespersec) # self.exposure, self.width, self.height, self.xoffset, self.yoffset,
        jai.Camera_Buffer.restype = ndpointer(dtype=np.uint8, shape=(height, width))
        jai.Camera_init(self.cam, self.configFile)
        
        self.width = width
        self.height = height
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.exposure = exposure
        
        self._defaultWidth = 2560
        self._defaultHeight = 2048
        self._defaultXoffset = 0
        self._defaultYoffset = 0
        self._defaultExposure = 8000
        jai.Camera_start(self.cam)
        
      
    def getImage(self):
        """
        This method get a frame on the selected camera and return a ndarray 
        """
        return jai.Camera_Buffer(self.cam)	
  
    def stop(self):
        """
        This method stops the acquisition 
        """
        jai.Camera_stop(self.cam)
        
    def close(self):
        """
        This method close properly the frame grabber
        It releases the allocated memory and stops the acquisition
        """
        return jai.Camera_close(self.cam)
    
    def restart(self):
        print "restart camera \n"
        jai.Camera_Buffer.restype = ndpointer(dtype=np.uint8, shape=(self.height, self.width))
        jai.Camera_start(self.cam)
        
    def reset_ZOI(self):
        print "stop camera \n"
        self.stop()
        self.cam = jai.Camera_new(self.numdevice, self.framespersec)
        jai.Camera_init(self.cam, self.configFile)
        self.yoffset = self._defaultYoffset
        self.xoffset = self._defaultXoffset
        self.height = self._defaultHeight
        self.width = self._defaultWidth
        self.restart()
        
    @property
    def height(self):
        print "height getter"
        return self._height
    
    @height.setter
    def height(self,height):
        print "height setter"
        jai.Camera_setHeight(self.cam, int(height))
        self._height=jai.Camera_getHeight(self.cam)
        
    @property
    def width(self):
        print "width getter"
        return self._width
    
    @width.setter
    def width(self,width):
        print "width setter"
        jai.Camera_setWidth(self.cam, (int(width)-(int(width)%32)))
        self._width= jai.Camera_getWidth(self.cam)

    @property
    def yoffset(self):
        print "yoffset getter"
        return self._yoffset
    
    @yoffset.setter
    def yoffset(self,yoffset):
        print "yoffset setter"
        jai.Camera_setYoffset(self.cam, int(yoffset))
        self._yoffset=jai.Camera_getYoffset(self.cam)
    
    @property
    def xoffset(self):
        print "xoffset getter"
        return self._xoffset

    @xoffset.setter
    def xoffset(self,xoffset):
        print "xoffset setter"
        jai.Camera_setXoffset(self.cam, (int(xoffset)-(int(xoffset)%32)))
        self._xoffset= jai.Camera_getXoffset(self.cam)
    
    @property
    def exposure(self):
        return self._exposure
    
    @property
    def gain(self):
        return None
        
    @exposure.setter
    def exposure(self, exposure):
        """
        this method changes the exposure of the camera
        set the exposure attribute and return a status (0 if it succed, -1 if it failed)
        """
        jai.Camera_setExposure(self.cam, int(exposure))
        self._exposure = jai.Camera_getExposure(self.cam)
              
    def __str__(self):
        """
        This method prints out the attributes values
        """
        return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} \n X offset: {5} \n Y offset: {6}".format(self.exposure, self.FPS, self.numdevice, self.width, self.height, self.xoffset, self.yoffset)
      