#include "ximea.h"

extern "C" {
    Camera* VideoCapture(int e_device){ return new Camera(e_device); }
    void *read(Camera* Camera){ Camera->xiread(); }
    void toString(Camera* Camera){ Camera->toString(); }
    bool isOpened(Camera* Camera){ Camera->isOpened(); }
    void release(Camera* Camera){ Camera->release(); }
    void addTrigger(Camera* Camera, int timeout, bool triggered){ Camera->addTrigger(timeout, triggered); }
    bool set(Camera* Camera, int propId, int value){ Camera->set(propId, value); }
    double get(Camera* Camera, int propId){ double test = Camera->get(propId); cout << "test : " << test << endl; return test;}
}