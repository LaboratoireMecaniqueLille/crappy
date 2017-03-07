/** @defgroup sources Sources
 *  This package contains source code for C or C++ libraries interfaced with Python. 
 *  @{
 */
/** @defgroup cameralink CameraLink
 *  CameraLink librarie to control camera devices through camera link interface.
 *  @{
 */

/** @defgroup capturecam CaptureCAM_CL
 *  CaptureCAM_CL class to control devices throught camera link interface.
 *  @{
 */

/** 
 * \file CameraLink.h
 * \brief CameraLink header, defines functions and classes for CameraLink module.
 * \author Robin Siemiatkowski
 * \version 0.1
 * \date 29/02/2016
 */
#ifndef CAMERA_LINK_H
#define CAMERA_LINK_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#ifndef WIN32
#ifdef _WIN32
#define WIN32 _WIN32
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep(x)
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <iostream>
#include <memory.h>
#include <string>
#include <typeinfo>
#include <map>
//#include <io.h>
// #include <unistd.h>
// #include <cstdint>
#include <fgrab_prototyp.h>
#include <SisoDisplay.h>
#define DLLEXPORT extern "C"
#define CHECK(param, paramDescr, Value)   if((Fg_setParameter(fg,param,Value, camPort)<0)){  \
					      sprintf(Data,"Fg_setParameter(%s) failed: %s\n",paramDescr, Fg_getLastErrorDescription(fg)); \
					      cout << "ERROR:" << Data << endl; \
					      throw string(Data);}
#define HandleResult(res,place) if (res!=FG_OK) {printf(" An error occured: %s (%d)\n",place,res);close();}

using namespace std;

class CaptureCAM_CL
{
public:
    CaptureCAM_CL(); /*!< Constructor */
    virtual ~CaptureCAM_CL(); /*!< Desctructor*/
    bool open(int index, const char* cameraType);
    void loadConfig(const char* conffile);
    void close();
    bool grabFrame();
    int startAcquire();
    int restart();
    virtual int stop();
    unsigned int getProperty(int);
    bool setProperty(int, int);
    void resetCvImage();
    void toString(); /*!< print out the attributes values of the cameralink instance. */
    int             format; /*!< Image format data */
    int             width; /*!< Image width*/
    int             height; /*!< Image height */
    int             xoffset; /*!< Image x offset */
    int             yoffset; /*!< Image y offset */
    int             boardNr; /*!< The number of the device */
    int             timeout; /*!< Timeout in microsecond */
    int             camPort; /*!< The number of the port on the cameraLink interface, should be PORT_A or PORT_B.*/
    double          framespersec; /*!< The number of frames per sec wanted*/
    unsigned int    exposure; /*!< Exposure time in microsecond*/
    const char*     cameraType; /*!< Type of camera, for frame grabber configuration */
    void            *ImgPtr; /*!< Pointer to image data*/
    bool isopened; /*!< State of the camera device*/
    bool isacquiring;
    char* serialWrite(char buffer[]);
    //void display(int dis);
private:
    void init();
    void errMsg(const char* msg, int errNum);
    frameindex_t    last_pic_nr;
    int nId;
    frameindex_t    cur_pic_nr;
    frameindex_t    nrOfPicturesToGrab;
    frameindex_t    nbBuffers;
    Fg_Struct       *fg;
    int             samplePerPixel;
    size_t          bytePerSample;
    unsigned int    TriggerMode;
    const char      *applet;
    dma_mem         *memHandle;
    char            Data[255];
    int             ComNr;
    void serialInit(unsigned int);
    void checkSerialCom(int);
    void *serialRefPtr;
};

// Defining videocapture struct
typedef struct {
  PyObject_HEAD;
  CaptureCAM_CL *camptr = NULL;
  const int device;
  const char* file;
  const char* camType;
} VideoCapture;

// Videocapture object prototypes
PyObject* VideoCapture_open(VideoCapture* self, PyObject *args);
PyObject* VideoCapture_isOpened(VideoCapture *self);
PyObject* VideoCapture_release(VideoCapture*);
PyObject* VideoCapture_read(VideoCapture *self);
PyObject* VideoCapture_get_array(VideoCapture *self);
PyObject* VideoCapture_set(VideoCapture *self, PyObject *args);
PyObject* VideoCapture_get(VideoCapture *self, PyObject *args);
PyObject* VideoCapture_startAcq(VideoCapture *self,PyObject *args);
PyObject* VideoCapture_stopAcq(VideoCapture *self);
PyObject* VideoCapture_serial_write(VideoCapture *self,PyObject *args);
PyObject* VideoCapture_load_config(VideoCapture *self, PyObject *args);

#endif
/** @} */
/** @} */
/** @} */
