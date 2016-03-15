#include "CameraLink.h"

DLLEXPORT {
    Camera* Camera_new(int e_boardNr,/* unsigned int e_exposure, unsigned int e_width, unsigned int e_height, unsigned int e_xoffset, unsigned int e_yoffset,*/ double e_framespersec){ return new Camera(e_boardNr, /*e_exposure, e_width, e_height, e_xoffset, e_yoffset,*/ e_framespersec); }
    int Camera_init(Camera* Camera, const char* file){ Camera->init(file); }
    void *Camera_Buffer(Camera* Camera){ Camera->getBuffer(); }
    void Camera_toString(Camera* Camera){ Camera->toString(); }
    void Camera_start(Camera* Camera){ Camera->startAcquire(); }
    void Camera_close(Camera* Camera){ Camera->close(); }
    void Camera_stop(Camera* Camera){ Camera->stop(); }
    void Camera_restart(Camera* Camera){ Camera->restart(); }
    unsigned int Camera_getExposure(Camera* Camera){Camera->getExposure();}
    void Camera_setExposure(Camera* Camera, unsigned int e_exposure){ Camera->setExposure(e_exposure); }
    
    unsigned int Camera_getWidth(Camera* Camera){Camera->getWidth();}
    void Camera_setWidth(Camera* Camera, unsigned int e_width){ Camera->setWidth(e_width); }
    
    unsigned int Camera_getHeight(Camera* Camera){Camera->getHeight();}
    void Camera_setHeight(Camera* Camera, unsigned int e_height){ Camera->setHeight(e_height); }
    
    unsigned int Camera_getXoffset(Camera* Camera){Camera->getXoffset();}
    void Camera_setXoffset(Camera* Camera, unsigned int e_xoffset){ Camera->setXoffset(e_xoffset); }
    
    unsigned int Camera_getYoffset(Camera* Camera){Camera->getYoffset();}
    void Camera_setYoffset(Camera* Camera, unsigned int e_yoffset){ Camera->setYoffset(e_yoffset); }
    
//     void Camera_serialWrite(Camera* Camera, char e_buf[]){ Camera-> serialWrite(e_buf); }
//     void Camera_serialInit(Camera* Camera, unsigned int serialIndex){ Camera-> serialInit(serialIndex); }   
}