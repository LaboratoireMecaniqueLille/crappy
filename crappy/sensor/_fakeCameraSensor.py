# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup fakecamerasensor FakeCameraSensor
# @{

## @file _fakeCameraSensor.py
# @brief  WIP. Fake camera sensor object
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016
import random
import time

from ._meta import cameraSensor
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    print "WARNING : OpenCV2 is not installed, some functionalities may crash"


class FakeCameraSensor(cameraSensor.CameraSensor):
    """WIP. Fake camera sensor object"""

    def __init__(self, numdevice=0, framespersec=None, external_trigger=False, data_format=0):
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
        self.it = self.frange(0, 10000, 1)

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
        # self.frame = plt.imread("/home/essais/Bureau/img_videoExtenso1.tiff")

    # @staticmethod
    # def frange(x, y, jump):
    #     a = x
    #     b = y
    #     while 1:
    #         if x < y:
    #             yield x
    #             x += jump
    #         else:
    #             if round(y) == float(a):
    #                 # yield y
    #                 x = a
    #                 y = b
    #             else:
    #                 yield y
    #                 y -= jump
    @staticmethod
    def frange(x, y, jump):
       for i in range(y):
           if x < y:
               yield x
               x += jump

    def get_image(self):
        """
        This method get a frame on the selected camera and return a ndarray
        If the camera breaks down, it reinitializes it, and tries again.
        """
        try:
            n, m = self.height, self.width
            Img = np.ones((n, m))
            [Y, X] = np.meshgrid(range(0, m), range(0, n))
            Y -= Y.max() / 2
            X -= X.max() / 2
            try:
                l0 = 100 + self.it.next() + random.randrange(-10, 10, 1) * random.random()
            except StopIteration:
                return Img.astype(np.uint8)
            r = 40
            Img[np.sqrt((X + l0 / 2) ** 2 + Y ** 2) < r] = 0
            Img[np.sqrt((X - l0 / 2) ** 2 + Y ** 2) < r] = 0
            return Img.astype(np.uint8)

        except Exception as e:
            print e
            #     frame = plt.imread(os.path.expanduser("~/Bureau/fake_camera_sensor_img.tiff"))
            # except IOError:
            #     try:
            #         frame = plt.imread(os.path.expanduser("~/Desktop/fake_camera_sensor_img.tiff"))
            #     except IOError:
            #         raise Exception("Path not found")

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

    @property
    def name(self):
        return "Dummy"

    def __str__(self):
        return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} \n X offset: {5} \n Y offset: {6}".format(
            self.exposure, self.FPS, self.numdevice, self.width, self.height, self.xoffset, self.yoffset)
