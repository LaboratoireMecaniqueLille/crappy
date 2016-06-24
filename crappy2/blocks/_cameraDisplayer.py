# coding: utf-8
from _meta import MasterBlock
from time import sleep,time


class CameraDisplayer(MasterBlock):
    """
    Simple images displayer. Can be paired with StreamerCamera
    """

    def __init__(self,framerate=5,cv=True,title='Displayer'):
        super(CameraDisplayer, self).__init__()
        self.delay = 1./framerate # Framerate (fps)
        self.cv = cv
        self.title = title
        print "cameraDisplayer!"

    def main(self):
        try:
            if not self.cv:
                import matplotlib.pyplot as plt
                plt.ion()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                first_loop = True
                while True:
                    # print "top loop"
                    frame = self.inputs[0].recv()
                    # print frame.shape
                    # if frame None:
                    # print frame[0][0]
                    im = plt.imshow(frame, cmap='gray')
                    plt.pause(0.001)
                    plt.show()
            else:
                import cv2
                data = 0
                cv2.namedWindow(self.title,cv2.WINDOW_NORMAL)
                while True:
                    t1 = time()
                    while data is not None: # To flush the pipe...
                        last = data
                        data = self.inputs[0].recv(False)
                    if last is not 0:
                        cv2.imshow('Displayer',last)
                        cv2.waitKey(1)
                    data = 0
                    while time()-t1 < self.delay:
                        data = self.inputs[0].recv()

        except (Exception, KeyboardInterrupt) as e:
            if self.cv:
                cv2.destroyAllWindows()
            else:
                plt.close('all')
            print "Exception in CameraDisplayer : ", e
