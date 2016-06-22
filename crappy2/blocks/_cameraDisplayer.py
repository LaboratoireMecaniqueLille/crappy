# coding: utf-8
from _meta import MasterBlock
from time import sleep 


class CameraDisplayer(MasterBlock):
    """
    Simple images displayer. Can be paired with StreamerCamera
    """

    def __init__(self,cv=True):
        super(CameraDisplayer, self).__init__()
        self.cv = cv
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
                for i in range(len(self.inputs)):
                    cv2.namedWindow('Displayer '+str(i+1),cv2.WINDOW_NORMAL)
                while True:
                    
                    for i in range(len(self.inputs)):
                        img = self.inputs[i].recv(False)
                        if img is not None:
                            cv2.imshow('Displayer '+str(i+1),img)
                            cv2.waitKey(1)

        except (Exception, KeyboardInterrupt) as e:
            if self.cv:
                cv2.destroyAllWindows()
            print "Exception in CameraDisplayer : ", e
            plt.close('all')
