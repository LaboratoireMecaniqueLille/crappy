import time
import crappy2.sensor.ximeaModule as xi
import cv2

ximea = xi.VideoCapture(0)
ximea.set(xi.CAP_PROP_XI_DATA_FORMAT, 0)  # 0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW
ximea.set(xi.CAP_PROP_XI_AEAG, 0)  # auto gain auto exposure
ximea.set(xi.CAP_PROP_FRAME_HEIGHT, 2048)
ximea.set(xi.CAP_PROP_EXPOSURE, 10000)
ximea.set(xi.CAP_PROP_FRAME_WIDTH, 2048)
ximea.set(xi.CAP_PROP_XI_OFFSET_Y, 0)
ximea.set(xi.CAP_PROP_XI_OFFSET_X, 0)


def stop():
    ximea.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow("Displayer")


def start():
    try:
        cv2.namedWindow("Displayer", cv2.WINDOW_NORMAL)
        ret, buf = ximea.read()
        if ret:
            cv2.imshow('Displayer', buf.get('data'))
            cv2.waitKey(1)
        else:
            print "failed to grab a frame"
        ximea.addTrigger(10000000, True)
    except Exception as e:
        print "exception: ", e


def show():
    ret, buf = ximea.read()
    if ret:
        cv2.imshow('Displayer', buf.get('data'))
        cv2.waitKey(1)
    else:
        print "fail\n"


def loop(i):
    j = 0
    while j < i:
        show()
        j += 1


def testPerf():
    t0 = time.time()
    i = 0
    while i < 200:
        ret, buf = ximea.read()
        if ret:
            cv2.imshow('Displayer', buf.get('data'))
            cv2.waitKey(1)
        else:
            print "fail"
        i += 1
    t1 = time.time()
    print "FPS:", 200 / (t1 - t0)
