# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup jaisensor Jai
# @{

## @file _jaiSensor.py
# @brief  Opens a Jai camera and allow to grab frame and set the various parameters.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

from os import path
import clModule as cl
import time

from ._meta import cameraSensor
import tkFileDialog as tk
from Tkinter import *


class Jai(cameraSensor.CameraSensor):
    def __init__(self, numdevice=0, framespersec=99, serial=None):
        """Opens a Jai camera and allow to grab frame and set the various parameters.
        
        Parameters
        ----------
        numdevice : int, dault = 0
            Number of the wanted device.
        framepersec : int, default = 99
            Wanted frame rate.
        """
        self.FPS = framespersec
        self.framespersec = framespersec
        self.numdevice = numdevice
        root = Tk()
        root.withdraw()
        self.configFile = tk.askopenfilename(parent=root)
        # self.configFile = "C:\\Users\\ECOLE\\fullareagray8.mcf"
        self.serial = serial
        self._init = True

    def new(self, exposure=8000, width=2560, height=2048, xoffset=0, yoffset=0, gain=None):
        """
        This method create a new instance of the camera jai class with the class attributes (device number, exposure, width, height, x offset, y offset and FPS)
        Then it prints these settings and initialises the camera and load a configuration file
        """
        self.jai = cl.VideoCapture(self.numdevice, self.configFile)

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
        self.jai.startAcq()

    def get_image(self):
        """
        This method get a frame on the selected camera and return a ndarray 
        If the camera breaks down, it reinitializes it, and tries again.
        """
        try:
            ret, frame = self.jai.read()
        except KeyboardInterrupt:
            print "KeyboardInterrupt, closing camera ..."
            self.close()
            self.quit = True

        try:
            if ret:
                return frame.get('data')
                # return frame
            elif not (self.quit):
                print "restarting camera..."
                time.sleep(0.5)
                # expo, wi, he, xoff,yoff,ga=self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain
                # self.reset()
                # self.__init__()
                # self.new(expo, wi, he, xoff,yoff,ga) # Reset the camera instance
                self.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset,
                         self.gain)  # Reset the camera instance
                return self.get_image()
        except UnboundLocalError:  # if ret doesn't exist, because of KeyboardInterrupt
            pass

    def stop(self):
        """This method stops the acquisition."""
        try:
            self.jai.stopAcq()
        except:
            print "cannot stop acquisition\n"

    def close(self):
        """This method close properly the frame grabber. 
        It releases the allocated memory and stops the acquisition.
        """
        try:
            self.jai.release()
        except Exception as e:
            print "cannot close Jai device: ", e

    def restart(self):
        """Restart the device."""
        self.jai.startAcq()

    def reset_zoi(self):
        """Reset to initial dimensions."""
        self.stop()
        self.yoffset = self._defaultYoffset
        self.xoffset = self._defaultXoffset
        self.height = self._defaultHeight
        self.width = self._defaultWidth
        self.restart()

    def set_zoi(self, width, height, xoffset, yoffset):
        """Define the Zone Of Interest"""
        self.stop()
        self.yoffset = yoffset
        self.xoffset = xoffset
        self.width = width
        self.height = height
        self.restart()

    @property
    def height(self):
        """Property. Set / get the current height"""
        return self._height

    @height.setter
    def height(self, height):
        self.jai.set(cl.FG_HEIGHT, int(height))
        self._height = self.jai.get(cl.FG_HEIGHT)
        if self.serial is not None:
            self.jai.serialWrite(self.serial.get_code(cl.FG_HEIGHT, self._height))

    @property
    def width(self):
        """Property. Set / get the current width"""
        return self._width

    @width.setter
    def width(self, width):
        self.jai.set(cl.FG_WIDTH, (int(width) - (int(width) % 32)))
        self._width = self.jai.get(cl.FG_WIDTH)
        if self.serial is not None:
            self.jai.serialWrite(self.serial.get_code(cl.FG_WIDTH, self._width))

    @property
    def yoffset(self):
        """Property. Set / get the current yoffset"""
        return self._yoffset

    @yoffset.setter
    def yoffset(self, yoffset):
        self.jai.set(cl.FG_YOFFSET, int(yoffset))
        self._yoffset = self.jai.get(cl.FG_YOFFSET)
        if (self.serial != None):
            self.jai.serialWrite(self.serial.get_code(cl.FG_YOFFSET, self._yoffset))

    @property
    def xoffset(self):
        """Property. Set / get the current xoffset"""
        return self._xoffset

    @xoffset.setter
    def xoffset(self, xoffset):
        self.jai.set(cl.FG_XOFFSET, (int(xoffset) - (int(xoffset) % 32)))
        self._xoffset = self.jai.get(cl.FG_XOFFSET)
        if (self.serial != None):
            self.jai.serialWrite(self.serial.get_code(cl.FG_XOFFSET, self._xoffset))

    @property
    def exposure(self):
        """Property. Set / get the current exposure
        Return a status (0 if it succed, -1 if it failed).
        """
        return self._exposure

    @property
    def gain(self):
        """No gain on Jai camera"""
        return None

    @exposure.setter
    def exposure(self, exposure):
        self.jai.set(cl.FG_EXPOSURE, int(exposure))
        self._exposure = self.jai.get(cl.FG_EXPOSURE)
        if self.serial is not None:
            self.jai.serialWrite(self.serial.get_code(cl.FG_EXPOSURE, self._exposure))

    def __str__(self):
        return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} " \
               "\n Height: {4} \n X offset: {5} \n Y offset: {6}".format(self.exposure, self.FPS, self.numdevice,
                                                                         self.width, self.height, self.xoffset,
                                                                         self.yoffset)

    @property
    def name(self):
        return "jai"
