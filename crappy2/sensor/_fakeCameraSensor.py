# coding: utf-8
from ._meta import cameraSensor
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print "WARNING : OpenCV2 is not installed, some functionalities may crash"


class FakeCameraSensor(cameraSensor.CameraSensor):
    """WIP. Fake camera sensor object"""

    def __init__(self, exposure, width, height, xoffset, yoffset, numdevice=0, framespersec=None,
                 external_trigger=False, data_format=0):
        super(FakeCameraSensor, self).__init__(numdevice, exposure, width, height, xoffset, yoffset, framespersec)
        self.quit = False
        self.FPS = framespersec
        self.framespersec = self.FPS
        self.numdevice = numdevice
        self.external_trigger = external_trigger
        self.data_format = data_format
        # self.actuator=None
        self._defaultWidth = 2048
        self._defaultHeight = 2048
        self._defaultXoffset = 0
        self._defaultYoffset = 0
        self._defaultExposure = 10000
        self._defaultGain = 0

    def new(self, exposure=10000, width=2048, height=2048, xoffset=0, yoffset=0, gain=0):
        """
        this method opens the ximea device. Ximea devices start at 1100. 1100 => device 0, 1101 => device 1
        And return a camera object
        """
        self.width = width
        self.height = height
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.exposure = exposure
        self.gain = gain

    def get_image(self):
        """
        This method get a frame on the selected camera and return a ndarray
        If the camera breaks down, it reinitializes it, and tries again.
        """
        try:
            frame = plt.imread(os.path.expanduser("~/Bureau/fake_camera_sensor_img.tiff"))
        except IOError:
            try:
                frame = plt.imread(os.path.expanduser("~/Desktop/fake_camera_sensor_img.tiff"))
            except IOError:
                raise Exception("Path not found")

    def close(self):
        """
        This method close properly the frame grabber
        It releases the allocated memory and stops the acquisition
        """
        print "closing camera..."

    def stop(self):
        pass

    def reset_ZOI(self):
        self.yoffset = self._defaultYoffset
        self.xoffset = self._defaultXoffset
        self.height = self._defaultHeight
        self.width = self._defaultWidth

    def set_ZOI(self, width, height, xoffset, yoffset):
        self.yoffset = yoffset
        self.xoffset = xoffset
        self.width = width
        self.height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        print "height setter : ", height
        self._height = ((int(height) / 2) * 2)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        print "width setter : ", width
        self._width = (int(width) - int(width) % 4)

    @property
    def yoffset(self):
        return self._yoffset

    @yoffset.setter
    def yoffset(self, yoffset):
        print "yoffset setter : ", yoffset
        y_offset = ((int(yoffset) / 2) * 2)
        self._yoffset = y_offset

    @property
    def xoffset(self):
        return self._xoffset

    @xoffset.setter
    def xoffset(self, xoffset):
        print "xoffset setter : ", xoffset
        x_offset = (int(xoffset) - int(xoffset) % 4)
        self._xoffset = x_offset

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, exposure):
        """
        this method changes the exposure of the camera
        and set the exposure attribute
        """
        print "exposure setter : ", exposure
        self._exposure = exposure

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        """
        this method changes the exposure of the camera
        and set the exposure attribute
        """
        print "gain setter : ", gain
        self._gain = gain

    def __str__(self):
        return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} \n X offset: {5} \n Y offset: {6}".format(
            self.exposure, self.FPS, self.numdevice, self.width, self.height, self.xoffset, self.yoffset)
