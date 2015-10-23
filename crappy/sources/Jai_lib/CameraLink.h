#ifndef DEF_CAMERA
#define DEF_CAMERA
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fgrab_prototyp.h>
#include <clser.h>
#define DLLEXPORT extern "C"
#define CHECK(param, paramDescr, Value)   if((Fg_setParameter(fg,param,Value, camPort)<0)){  \
					      sprintf(Data,"Fg_setParameter(%s) failed: %s\n",paramDescr, Fg_getLastErrorDescription(fg)); \
					      cout << "ERROR:" << Data << endl; \
					      throw string(Data);}
using namespace std;
extern "C"{
class Camera
{ 
    private:
    
    frameindex_t last_pic_nr;
    frameindex_t cur_pic_nr;
    int timeout;
    Fg_Struct *fg;
    int m_boardNr;
    int camPort;
    frameindex_t nrOfPicturesToGrab;
    frameindex_t nbBuffers;
    int samplePerPixel;
    size_t bytePerSample;
    unsigned int TriggerMode;
    const char *applet;
    dma_mem *memHandle;
    void *serialRefPtr;
    unsigned int m_width;
    unsigned int m_height;
    double m_framespersec;
    unsigned int m_xoffset;
    unsigned int m_yoffset;
    unsigned int m_exposure;
    char Data[255];
    void serialInit(unsigned int);
    int checkSerialCom(int);
    const char* m_file;
    int ComNr;
    
    public:
    
    Camera(int ,/* unsigned int , unsigned int , unsigned int , unsigned int , unsigned int ,*/ double );
    ~Camera();
    void toString();
    int init(const char *);
    int startAcquire();
    int restart();
    void close();
    void stop();
    int setExposure(unsigned int);
    int setWidth(unsigned int);
    int setHeight(unsigned int);
    int setXoffset(unsigned int);
    int setYoffset(unsigned int);
    unsigned int getExposure();
    unsigned int getWidth();
    unsigned int getHeight();
    unsigned int getXoffset();
    unsigned int getYoffset();
    void *getBuffer();
    void serialWrite(char buffer[]);
};
}

#endif