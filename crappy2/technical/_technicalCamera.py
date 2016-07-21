# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup TechnicalCamera TechnicalCamera
# @{

## @file _technicalCamera.py
# @brief Opens a camera device and initialise it (with cameraInit found in crappy[2]/technical)
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

from multiprocessing import Process, Pipe
from . import get_camera_config


class TechnicalCamera(object):
    """
    Opens a camera device and initialise it.
    """

    def __init__(self, camera="ximea", num_device=0, videoextenso=None):
        """
        This Class opens a device and runs the initialisation sequence (CameraInit).

        It then closes the device and keep the parameters in memory for later use.
        Args:
            camera : {'ximea','jai'}, default = 'ximea'
                Name of the desired camera device.
            num_device : int, default = 0
                Number of the desired device.
            videoextenso : dict
            dict of parameters that you can use to pass informations.

            * 'enabled' : Bool
                Set True if you need the videoextenso.
            * 'white_spot' : Bool
                Set to True if your spots are white on a dark material.
            * 'border' : int, default = 4
                Size of the border for spot detection
            * 'x_offset' : int
                Offset for the x-axis.
            * 'y_offset' : int
                Offset for the y-axis
            * 'height' : int
                Height of the image, in pixels.
            * 'width : int
                Width of the image, in pixels.
        """
        if videoextenso is None:
            videoextenso = {}
        try:
            module = __import__("crappy2.sensor", fromlist=[camera.capitalize()])
            camera_class = getattr(module, camera.capitalize())
        except Exception as e:
            print "{0}".format(e), " : Unreconized camera\n"
            import sys
            sys.exit()
        try:
            module = __import__("crappy2.sensor.clserial", fromlist=[camera.capitalize() + "Serial"])
            code_class = getattr(module, camera.capitalize() + "Serial")
            from crappy2.sensor.clserial import ClSerial as cl
            ser = code_class()
            self.serial = cl(ser)
        except ImportError:
            self.serial = None
        except Exception as e:
            print "{0}".format(e)
            self.serial = None
        # print "module, camera_class, serial : ", module, camera_class, self.serial
        # initialisation:
        self.sensor = camera_class(numdevice=num_device)
        self.video_extenso = videoextenso
        recv_pipe, send_pipe = Pipe()
        print "lauching camera config..."
        proc_test = Process(target=get_camera_config, args=(self.sensor, self.video_extenso, send_pipe))
        proc_test.start()
        data = recv_pipe.recv()
        print "data received, config done."
        if self.video_extenso['enabled']:
            self.exposure, self.gain, self.width, self.height, self.x_offset, self.y_offset, self.minx, self.max_x, \
                self.miny, self.maxy, self.NumOfReg, self.L0x, self.L0y, self.thresh, self.Points_coordinates = data[:]
        else:
            self.exposure, self.gain, self.width, self.height, self.x_offset, self.y_offset = data[:]

        # here we should modify height, width and others in camera_class().sensor #WIP
        # self.sensor.exposure=self.exposure
        # self.sensor.gain=self.gain
        proc_test.terminate()
        # self.cam.new(exposure=exposure, gain=gain, y_offset=y_offset, x_offset=x_offset, height=height, width=width)

        # def _interface(self, send_pipe, camera):
        # settings = getCameraConfig(camera, self.videoextenso)
        # send_pipe.send(settings)

    def __str__(self):
        return self.sensor.__str__()
