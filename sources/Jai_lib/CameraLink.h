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

#ifndef XIMEA_H
#define XIMEA_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"

#ifndef WIN32
#ifdef _WIN32
#define WIN32 _WIN32
#include <windows.h>
#define sleep(x) Sleep(x)
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include <memory.h>
#include <string.h>
#include <io.h>
// #include <unistd.h>
#include <typeinfo>
// #include <cstdint>
#include <fgrab_prototyp.h>
#include <clser.h>
// #include "export.h"
#include <numpy/arrayobject.h>
#include <SisoDisplay.h>
// #include <datetime.h>
#include "structmember.h"
// #include <map>
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
    virtual bool open( int index, const char* file);
    virtual void close();
    virtual bool grabFrame();
    int startAcquire();
    int restart();
    int stop();
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
    const char*     file; /*!< path to the configuration file of the camera*/
    void            *ImgPtr; /*!< Pointer to image data*/
    bool isopened; /*!< State of the camera device*/
    void serialWrite(char buffer[]);
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
    int checkSerialCom(int);
    void *serialRefPtr;
};


extern "C" {
    typedef struct {
        PyObject_HEAD
        PyObject *myarray;
        int device;
        const char* file;
    } VideoCapture;
    //PyObject* VideoCapture_Display(VideoCapture *self, PyObject *args);
    PyObject* VideoCapture_open(int device, const char* file);
    PyObject* VideoCapture_isOpened();
    PyObject* VideoCapture_release();
    PyObject* VideoCapture_startAcquisition();
    PyObject* VideoCapture_stopAcquisition();
//     PyObject* VideoCapture_addTrigger(VideoCapture *self, PyObject *args);
    bool VideoCapture_grab();
    PyObject* VideoCapture_retrieve(VideoCapture *self);
    PyObject* VideoCapture_fgread(VideoCapture *self);
    PyObject* VideoCapture_set(VideoCapture *self, PyObject *args);
    PyObject* VideoCapture_get(VideoCapture *self, PyObject *args);
    PyObject* VideoCapture_getMeta();
    PyObject* VideoCapture_serialSet(VideoCapture *self, PyObject *args);
}
#endif
/** @} */ 
/** @} */
/** @} */