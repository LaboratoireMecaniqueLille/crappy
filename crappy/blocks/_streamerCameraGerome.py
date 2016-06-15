# coding: utf-8
from _meta import MasterBlock
import os
import time
from crappy.technical import TechnicalCamera as tc
from multiprocessing import Pipe, Process
from ..links._link import TimeoutError


class CameraReader:
    def __init__(self, cam, exposure, width, height, xoffset, yoffset, gain):
        self.cam = cam
        self.exposure = exposure
        self.width = width
        self.height = height
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.gain = gain
        self.parent, self.child = Pipe()
        self.proc = Process(target=self.start, args=(self.child,))
        self.proc.start()

    def start(self, child):
        self.cam.sensor.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain)
        while 1:
            image = self.cam.sensor.get_image()
            if child.poll():
                data = child.recv()
                if data == "break":
                    break
                child.send(image)
        self.cam.sensor.close()

    def get_image(self):
        self.parent.send('ok')
        image = self.parent.recv()
        return image

    def close(self):
        self.parent.send("break")
        self.proc.join()


class StreamerCameraG(MasterBlock):
    """
    Streams pictures.
    """

    def __init__(self, camera, numdevice=0, freq=None, save=False, save_directory="./images/", label="cycle", xoffset=0,
                 yoffset=0, width=2048, height=2048):
        """
        This block fetch images from a camera object, can save and/or transmit them to
        another block. It can be triggered by a Link or internally
        by defining the frequency.

        Parameters
        ----------
        camera : {"Ximea","Jai"}
            See sensor.cameraSensor documentation.
        numdevice : int, default = 0
            If you have several camera plugged, choose the right one.
        freq : float or int or None, default=None
            Wanted acquisition frequency. Cannot exceed acquisition device capability.
            If None, will go as fast as possible.
        save : boolean, default =False
            Set to True if you want to save images.
        save_directory : directory, default = "./images/"
            directory to the saving folder. If inexistant, will be created.
        label : string, default="cycle"
            label of the input data you want to save in the name of the saved image, in
            case of external trigger.
        xoffset: int, default = 0
            Offset on the x axis.
        yoffset: int, default = 0
            Offset on the y axis.
        width: int, default = 2048
            Width of the image.
        height: int, default = 2048
            Height of the image.
        """
        super(StreamerCameraG, self).__init__()
        print "streamer camera!!"
        import SimpleITK as sitk
        self.sitk = sitk
        self.numdevice = numdevice
        print 'test1'
        self.camera = tc(camera, self.numdevice,
                         videoextenso={'enabled': False, 'xoffset': xoffset, 'yoffset': yoffset, 'width': width,
                                       'height': height})
        print 'test2'
        self.freq = freq
        self.save = save
        self.i = 0
        self.save_directory = save_directory
        self.label = label
        self.width = self.camera.width
        self.height = self.camera.height
        self.xoffset = self.camera.x_offset
        self.yoffset = self.camera.y_offset
        self.exposure = self.camera.exposure
        self.gain = self.camera.gain
        if not os.path.exists(self.save_directory) and self.save:
            os.makedirs(self.save_directory)

    def main(self):
        print "streamer camera!!", os.getpid()
        self.camera = CameraReader(self.camera, self.exposure, self.width, self.height, self.xoffset, self.yoffset,
                                   self.gain)
        try:
            _a = self.inputs[:]
            trigger = "external"
        except AttributeError:
            trigger = "internal"
        timer = time.time()
        try:
            print "start :", time.time() - self.t0
            while True:
                if trigger == "internal":
                    if self.freq is not None:
                        while time.time() - timer < 1. / self.freq:
                            pass
                    timer = time.time()
                    img = self.camera.get_image()
                    if self.save:
                        image = self.sitk.GetImageFromArray(img)
                        self.sitk.WriteImage(image,
                                             self.save_directory + "img_%.6d.tiff" % (self.i))
                        self.i += 1
                elif trigger == "external":
                    Data = self.inputs[0].recv()  # wait for a signal
                    if Data is not None:
                        img = self.camera.get_image()
                        t = time.time() - self.t0
                        if self.save:
                            image = self.sitk.GetImageFromArray(img)
                            try:
                                self.sitk.WriteImage(image,
                                                     self.save_directory + "img_%.6d_cycle%09.1f.tiff" % (
                                                         self.i, Data[self.label]))
                            except KeyError:
                                self.sitk.WriteImage(image,
                                                     self.save_directory + "img_%.6d.tiff" % self.i)
                            self.i += 1
                try:
                    if trigger == "internal" or Data is not None:
                        for output in self.outputs:
                            output.send(img)
                except TimeoutError:
                    raise
                except AttributeError:  # if no outputs
                    pass

        except (Exception, KeyboardInterrupt) as e:
            print "Exception in streamerCamera : ",
            self.camera.close()
            # raise
