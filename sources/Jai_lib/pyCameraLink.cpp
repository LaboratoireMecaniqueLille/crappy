/** @addtogroup sources
 *  @{
 */

/** @addtogroup cameralink 
 *  @{
 */

/** @addtogroup videocapture 
 *  @{
 */

/** 
 * \file pyCameraLink.cpp
 * \brief CameraLink class to be interfaced with Python
 * \author Robin Siemiatkowski
 * \version 0.1
 * \date 29/02/2016
 */

#include "CameraLink.h"
#include "export.h"
#include <numpy/ndarrayobject.h>
#include <datetime.h>
#include "structmember.h"
CaptureCAM_CL* capt;
PyObject *myDict = PyDict_New();
PyObject *rslt = PyTuple_New(2);
char *array_buffer;

#ifdef __cplusplus
extern "C" {
#endif 

/**
 * \fn PyObject* VideoCapture_open(int device, const char* file)

 * \brief 
 *      method to open a camera device through the cameraLink interface.
 * \param device The number of the device to be openned.
 * \return a call to VideoCapture_isOpened()
 */
PyObject* VideoCapture_open(int device, const char* file)
{
    if (VideoCapture_isOpened() == Py_True) {
		VideoCapture_release();
	}
	capt->open(device, file);
    return VideoCapture_isOpened();
}

/*
 * \fn PyObject* VideoCapture_addTrigger(VideoCapture *self, PyObject *args)
 * \brief 
 *      method to open a camera device through the cameraLink interface.
 * \param device The number of the device to be openned.
 * \return a call to VideoCapture_isOpened()
 */
/*
PyObject*
VideoCapture_addTrigger(VideoCapture *self, PyObject *args)
{
    int timeout;
    bool triggered;
    if (!PyArg_ParseTuple(args, "ib", &timeout, &triggered))
            exit(0);
    capt->addTrigger(timeout, triggered);
    return Py_None;
}*/



/**
 * \fn void VideoCapture_startAcquisition()
 * \brief 
 *      Start the acquisition of a camera device.
 * 	The acquisition is started automaticaly after a call to the constructor of the VideoCapture class.
 * 	If you want to change some parameters of the acquisition, you have to call stopAcq() first.
 */
PyObject* VideoCapture_startAcquisition()
{
    if(!capt->startAcquire())
      return Py_True;
    else
      return Py_False;
}

/**
 * \fn void VideoCapture_stopAcquisition()
 * \brief 
 *      Stop the acquisition of a camera device.
 * 	The acquisition is started automaticaly after a call to the constructor of the VideoCapture class.
 * 	If you want to change some parameters of the acquisition, you have to call stopAcq() first.
 */
PyObject* VideoCapture_stopAcquisition()
{
   if(!capt->stop())
     return Py_True;
   else 
     return Py_False;
}

/**
 * \fn PyObject* VideoCapture_isOpened()
 * \brief 
 *      It check if te camera is running or it has been stopped.
 * \return 
 *      - Py_True if the camera is openned.
 *      - Py_False if the camera is closed.
 */
PyObject* VideoCapture_isOpened()
{
    if(capt->isopened) {
		return Py_True;
	}else{
		return Py_False;
	}
}

/**
 * \fn PyObject* VideoCapture_release()
 * \brief 
 *      Close the camera device: stop the acquisition and free the allocated memory.
 */
PyObject* VideoCapture_release()
{
    capt->close();
    return Py_None;
}

/**
 * \fn PyObject* VideoCapture_serialSet(VideoCapture *self, PyObject *args)
 * \brief Setting the value of a parameter from the serial interface 
 *
 * \param buffer will be written to the serial port.
 */
PyObject* VideoCapture_serialSet(VideoCapture *self, PyObject *args)
{
	char *buffer ;
	if (!PyArg_ParseTuple(args, "s", &buffer))
		exit(0);
	capt->serialWrite(buffer);
	return Py_None;
  
}




/**
 * \fn bool VideoCapture_grab()
 * \brief Stop the acquisition and free the allocated memory.
 * \return True if the frame grabber has successfully grabbed a frame.
 */
bool VideoCapture_grab()
{
    return capt->grabFrame();
}

/**
 * \fn PyObject* VideoCapture_retrieve(VideoCapture *self)
*  \brief Method to retrieve a frame to Python understanding format.
*  It get the pointer to previously grabbed frame, and convert it to a numpy array object.
 * \return 
 *      - myDict, which contains the frame's data and the meta data of the grabbed frame.
 *      - Py_None if the current format has not been retrived.
 */
PyObject* VideoCapture_retrieve(VideoCapture *self)
{

        if(array_buffer==NULL){
            free(array_buffer);
            array_buffer=NULL;
        }
		switch(capt->format)
		{
		case FG_BINARY: {
			cout << "FG_BINARY" << endl;
			const int ndim = 2;
			npy_intp nd[2] = {capt->height, capt->width};
                        Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
                        Py_XDECREF(nd);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->ImgPtr, capt->width*capt->height);
			break;}
		case FG_GRAY:{
			const int ndim = 2;
			npy_intp nd[2] = {capt->height, capt->width};
            Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNewFromData(ndim, nd, NPY_UINT8, capt->ImgPtr);
			Py_XDECREF(nd);
			break;
			}
		case FG_GRAY16: {
			cout << "FG_GRAY16" << endl;
			const int ndim = 2;
			npy_intp nd[2] = {capt->height,capt->width};
            Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNewFromData(ndim, nd, NPY_UINT16, capt->ImgPtr);
			Py_XDECREF(nd);
			break;
			}
		case FG_COL24: {
			cout << "FG_COL24" << endl;
			const int ndim = 1;
			npy_intp nd[1] = {capt->width*capt->height*3};
                        Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT16);
                        Py_XDECREF(nd);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->ImgPtr, capt->width*capt->height*3);
			break;}
		case FG_COL30: {
			const int ndim = 4;
			npy_intp nd[3] = {capt->height, capt->width, 4};
                        Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
                        Py_XDECREF(nd);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->ImgPtr, capt->width*capt->height*4);
			break;}
		case FG_COL32  : {
			const int ndim = 3;
			npy_intp nd[3] = {capt->height, capt->width, 3};
                        Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
                        Py_XDECREF(nd);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->ImgPtr, capt->width*capt->height*3);
			break;}
		case FG_COL48: {
			const int ndim = 2;
			npy_intp nd[2] = {capt->height, capt->width};
                        Py_XDECREF(self->myarray);
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
                        Py_XDECREF(nd);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->ImgPtr, capt->width*capt->height);
			break;}
		default : 
			return Py_None;
		}
		//capt-> resetCvImage();
        myDict = PyDict_New();
		myDict = VideoCapture_getMeta();
		PyDict_SetItemString(myDict, "data", self->myarray);
    return myDict;
}

/**
 * \fn PyObject* VideoCapture_getMeta()
*  \brief Method to get meta data of the grabbed frame.
 * \return 
 *      - myDict, which contains the meta data of the grabbed frame:
 *          -# the width of the grabbed frame
 *          -# the height of the grabbed frame
 */
PyObject* VideoCapture_getMeta()
{
// 	PyDateTime_IMPORT;
	PyDict_SetItemString(myDict, "width", Py_BuildValue("I", capt->width));
	PyDict_SetItemString(myDict, "height", Py_BuildValue("I",capt->height));
// 	PyDict_SetItemString(myDict, "nframe", Py_BuildValue("I",capt->image.nframe));
// 	PyDict_SetItemString(myDict, "AbsoluteOffsetX", Py_BuildValue("I",capt->image.AbsoluteOffsetX));
// 	PyDict_SetItemString(myDict, "AbsoluteOffsetY", Py_BuildValue("I",capt->image.AbsoluteOffsetY));
// 	
// 	PyObject *floatObj = PyFloat_FromDouble(capt->image.tsSec);
// 	PyObject *timeTuple = Py_BuildValue("(O)", floatObj);  
// 	PyObject *dateTime = PyDateTime_FromTimestamp(timeTuple);
// 	PyDict_SetItemString(myDict, "tsSec", dateTime);
//         Py_CLEAR(floatObj);
//         Py_CLEAR(timeTuple);
//         Py_CLEAR(dateTime);
// 	
// 	PyObject *floatObj1 = PyFloat_FromDouble(capt->image.tsUSec);
// 	PyObject *timeTuple1 = Py_BuildValue("(O)", floatObj1);  
// 	PyObject *dateTime1 = PyDateTime_FromTimestamp(timeTuple1);
// 	PyDict_SetItemString(myDict, "tsUSec", dateTime1);
// 	Py_CLEAR(floatObj1);
//      Py_CLEAR(timeTuple1);
//      Py_CLEAR(dateTime1);
        
	return myDict;
}

/**
 * \fn PyObject* VideoCapture_fgread(VideoCapture *self)
*  \brief Method to get the dictionnary containing the meta data, the frame data and the status of the grabbed frame.
 * \return A PyTuple containing the status of the grabbing as first position:
 *          - Py_False if the grabbing has failled
 *          - Py_True if the grabbing has succeded 
 *         and the disctionnary containing the data and meta data of the grabbed frame, or Py_None if the grabbing has failled.
 */
PyObject* VideoCapture_fgread(VideoCapture *self)
{
	rslt = PyTuple_New(2);
	if(!VideoCapture_grab()){
		PyTuple_SetItem(rslt, 0, Py_False);
		PyTuple_SetItem(rslt, 1, Py_None);
    }else{
		PyTuple_SetItem(rslt, 0, Py_True);
		PyTuple_SetItem(rslt, 1, VideoCapture_retrieve(self));
		//PyTuple_SetItem(rslt, 1, Py_BuildValue("O&", capt->ImgPtr));
    }
    return rslt;
    
}

/**
 * \fn PyObject* VideoCapture_set(VideoCapture *self, PyObject *args)
 * \brief Setting the value of a parameter from a frame grabber. 
 *
 * \param property_id As argument, a identification number is needed.
    If the identification number is unknown, the parameter name has to be given.
    It can be one of the following:
        - FG_TIMEOUT: Time in seconds until device driver displays a timeout of the frame grabber. 
        - FG_WIDTH: Width of the clipping image.
        - FG_HEIGHT: Height of the clipping image.
        - FG_XSHIFT: Number of invalid words at the beginning of a row (modulo of the width of the interface).
        - FG_XOFFSET: X-offset from the left top corner in pixel.
        - FG_YOFFSET: Y-offset from the left top corner in pixel. 
        - FG_FRAMESPERSEC: Number of images per second.
        - FG_EXPOSURE: Exposure time in µs.
        - FG_FORMAT: Color format of the transferred image
                        -# 8bit gray (FG_GRAY)
                        -# 16bit color (FG_GRAY16)
                        -# 24bit color (FG_COL24).
                     See color management of the according frame grabber design. 
        - FG_PORT: Logical number of the active CameraLink port.
        - FG_TRIGGERMODE: Trigger modes:
                            -# FREE_RUN
                            -# GRABBER_CONTROLLED
                            -# GRABBER_CONTROLLED_SYNCRON
                            -# ASYNC_SOFTWARE_TRIGGER
                            -# ASYNC_TRIGGER. 
        - FG_STROBPULSEDELAY: Strobe delay to the trigger in µs.
        - FG_GLOBAL_ACCESS: Returns the value for the set plausibility access.
        
 * \param value  Pointer to required value.
 * \return Py_True if the parameter was read correctly
           Py_False if an invalid parameter has been entered or if the entered value is besides valid ranges.
 */
PyObject* VideoCapture_set(VideoCapture *self, PyObject *args)
{
	int propId;
	int value;
	if (!PyArg_ParseTuple(args, "ii", &propId, &value))
		exit(0);
    if(capt->setProperty(propId, value)) return Py_True;
	else return Py_False;
}

/**
 * \fn PyObject* VideoCapture_get(VideoCapture *self, PyObject *args)
 * \brief Reading the current value of a parameter from a frame grabber. 
 *
 * \param property_id As argument, a identification number is needed.
 *   If the identification number is unknown, the parameter name has to be given.
 *   It can be one of the following:
 *       - FG_CAMSTAUS: If a camera signal is on CameraLink port value is 1 else 0.
 *       - FG_REVNR: Current revision version of camera DLL.
 *       - FG_TIMEOUT: Time in seconds until device driver displays a timeout of the frame grabber. 
 *       - FG_WIDTH: Width of the clipping image.
 *       - FG_MAXWIDTH: Maximum width of the clipping image.
 *       - FG_HEIGHT: Height of the clipping image.
 *       - FG_MAXHEIGHT: Maximum height of the clipping image.
 *       - FG_XSHIFT: Number of invalid words at the beginning of a row (modulo of the width of the interface).
 *       - FG_XOFFSET: X-offset from the left top corner in pixel.
 *       - FG_YOFFSET: Y-offset from the left top corner in pixel. 
 *       - FG_FRAMESPERSEC: Number of images per second.
 *       - FG_EXPOSURE: Exposure time in µs.
 *       - FG_FORMAT: Color format of the transferred image
 *                       -# 8bit gray (FG_GRAY)
 *                       -# 16bit color (FG_GRAY16)
 *                       -# 24bit color (FG_COL24).
 *                    See color management of the according frame grabber design. 
 *       - FG_PORT: Logical number of the active CameraLink port.
 *       - FG_PIXELDEPTH: Returns the depth of color of the pixel.
 *       - FG_LINEALIGNMENT: Returns the alignment of a line (in bits).
 *       - FG_TRANSFER_LEN: Returns the length of the last DMA transfer.
 *       - FG_TRIGGERMODE: Trigger modes:
 *                           -# FREE_RUN
 *                           -# GRABBER_CONTROLLED
 *                           -# GRABBER_CONTROLLED_SYNCRON
 *                           -# ASYNC_SOFTWARE_TRIGGER
 *                           -# ASYNC_TRIGGER. 
 *       - FG_STROBPULSEDELAY: Strobe delay to the trigger in µs.
 *       - FG_TWOCAMMODEL: Returns the value, if the loaded camera applet is a dual (1) or a single applet (0).
 *       - FG_HDSYNC: Returns the HDSYNC value. 
 *       - FG_GLOBAL_ACCESS: Returns the value for the set plausibility access.
 *       - FG_BOARD_INFORMATION: Information on the board type:
 *                                   -# BINFO_BOARD_TYPE:
 *                                       - 0xa40 for microEnable IV-Base x1
 *                                       - 0xa41 for microEnable IV-Full x1
 *                                       - 0xa44 for microEnable IV-Full x4
 *
 *                                   -# BINFO_POCL:
 *                                       - 0 for microEnable IV-Base x1
 *                                       - 1 for microEnable IV-Base x1 PoCL
 * \return 
 *       - Py_True if the parameter was read correctly
 *       - Py_False if an invalid parameter has been entered.
 */
PyObject* VideoCapture_get(VideoCapture *self, PyObject *args)
{
	int propId;
	if (!PyArg_ParseTuple(args, "i", &propId))
		exit(0);
    return Py_BuildValue("i", capt->getProperty(propId));
}

/**
 * \fn static void VideoCapture_dealloc(VideoCapture* self)
 *  \brief The destructor of the VideoCapture class
 *   It release the camera device, and free all the allocated memory.
 */
static void VideoCapture_dealloc(VideoCapture* self)
{
    Py_XDECREF(self->myarray);
    VideoCapture_release();
    self->ob_type->tp_free((PyObject*)self);
}

/**
 * \fn static PyObject * VideoCapture_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
 * \brief The constructor of the VideoCapture class
 * It create the self parameter to be use later in Python.
 * \return self parameter, which represent an instance of the VideoCapture class. 
 */
static PyObject * VideoCapture_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    VideoCapture *self;

    self = (VideoCapture *)type->tp_alloc(type, 0);
    if (self != NULL) {
		if (!PyArg_ParseTuple(args, "is:call", &self->device, &self->file)) {
			return NULL;
		}
    }
    return (PyObject *)self;
}

/**
 * \fn static int VideoCapture_init(VideoCapture *self, PyObject *args, PyObject *kwds)
 * \brief This method is called after the constructor, it initialize the VideoCapture instance
 * It opens the camera device and create a CaptureCAM_CL instance.
 * \return 0 if the parsed argument are valable, -1 if not.
 */
static int VideoCapture_init(VideoCapture *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"device","file",  NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|is", kwlist,  
                                      &self->device, &self->file))
        return -1; 
    capt = new CaptureCAM_CL();
    VideoCapture_open(self->device, self->file);
    return 0;
}

/*
PyObject* VideoCapture_Display(VideoCapture *self, PyObject *args)
{
	int open;
	if (!PyArg_ParseTuple(args, "i", &open)){
// 		exit(0);
	    cout << "can't read argument" << endl;
	}else{
	    capt->display(open);
	}
	return Py_None;
}*/

static PyMemberDef VideoCapture_members[] = {
    {NULL} 
};


// only for windows
void set_map_to_export(){
	my_map.insert(make_pair("FG_OK", FG_OK));// The parameter was set correctly.
	my_map.insert(make_pair("FG_NOT_INIT ",  FG_NOT_INIT));// Initialization failed.
	my_map.insert(make_pair("FG_INVALID_PARAMETER", FG_INVALID_PARAMETER));// An invalid parameter has been entered.
	my_map.insert(make_pair("FG_VALUE_OUT_OF_RANGE" , FG_VALUE_OUT_OF_RANGE));// Value is besides valid ranges.
	my_map.insert(make_pair("FG_CAMSTATUS", FG_CAMSTATUS));// If a camera signal is on CameraLink port value is 1 else 0.
	my_map.insert(make_pair("FG_REVNR ", FG_REVNR));// Current revision version of camera DLL.
	my_map.insert(make_pair("FG_TIMEOUT", FG_TIMEOUT));// Time in seconds until device driver displays a timeout of the frame grabber.
	my_map.insert(make_pair("FG_WIDTH", FG_WIDTH));//  Width of the clipping image.
	my_map.insert(make_pair("FG_MAXWIDTH", FG_MAXWIDTH));// Maximum width of the clipping image.
	my_map.insert(make_pair("FG_HEIGHT", FG_HEIGHT));// Height of the clipping image.
	my_map.insert(make_pair("FG_MAXHEIGHT", FG_MAXHEIGHT));// Maximum height of the clipping image.
	my_map.insert(make_pair("FG_XSHIFT", FG_XSHIFT));// Number of invalid words at the beginning of a row (modulo of the width of the interface). 
	my_map.insert(make_pair("FG_XOFFSET", FG_XOFFSET));// X-offset from the left top corner in pixel. 
	my_map.insert(make_pair("FG_YOFFSET", FG_YOFFSET));// Y-offset from the left top corner in pixel. 
	my_map.insert(make_pair("FG_FRAMESPERSEC", FG_FRAMESPERSEC));// Number of images per second.
	my_map.insert(make_pair("FG_MAXFRAMESPERSEC", FG_MAXFRAMESPERSEC));// Max number of images per second.
	my_map.insert(make_pair("FG_EXPOSURE", FG_EXPOSURE));// Exposure time in µs.
        /*
         * Color format of the transferred image: 
         *  1bit (FG_BINARY),
         *  8bit (FG_GRAY),
         *  16bit (FG_GRAY16),
         *  24bit (FG_COL24),
         *  30bit (FG_COL30),
         *  32bit (FG_COL32),
         *  48bit (FG_COL48).
         * See color management of the according frame grabber design.  
         */
	my_map.insert(make_pair("FG_FORMAT", FG_FORMAT));
        
	my_map.insert(make_pair("FG_BINARY", FG_BINARY));
	my_map.insert(make_pair("FG_GRAY", FG_GRAY));
	my_map.insert(make_pair("FG_GRAY16 ", FG_GRAY16));
	my_map.insert(make_pair("FG_COL24", FG_COL24));
	my_map.insert(make_pair("FG_COL30", FG_COL30));
	my_map.insert(make_pair("FG_COL32", FG_COL32));
	my_map.insert(make_pair("FG_COL48 ", FG_COL48));
	
	my_map.insert(make_pair("PORT_A", PORT_A)); // base config
	my_map.insert(make_pair("PORT_B", PORT_B)); // base config
	my_map.insert(make_pair("FG_COL48 ", PORT_AB)); // medium/full config
        
	my_map.insert(make_pair("FG_PORT", FG_PORT));//Logical number of the active CameraLink port.
	my_map.insert(make_pair("FG_PIXELDEPTH", FG_PIXELDEPTH));// Returns the depth of color of the pixel. 
	my_map.insert(make_pair("FG_LINEALIGNMENT ", FG_LINEALIGNMENT));// Returns the alignment of a line (in bits).
	my_map.insert(make_pair("FG_RIGHT_ALIGNED", FG_RIGHT_ALIGNED)); // right
	my_map.insert(make_pair("FG_LEFT_ALIGNED ", FG_LEFT_ALIGNED)); // left
	
	my_map.insert(make_pair("FG_TRANSFER_LEN", FG_TRANSFER_LEN));// Returns the length of the last DMA transfer.
	
        /* 
         *Trigger modes: 
         *  - FREE_RUN
         *  - GRABBER_CONTROLLED
         *  - GRABBER_CONTROLLED_SYNCRON
         *  - ASYNC_SOFTWARE_TRIGGER
         *  - ASYNC_TRIGGER 
         */
	my_map.insert(make_pair("FG_TRIGGERMODE", FG_TRIGGERMODE));
        
	my_map.insert(make_pair("FREE_RUN", FREE_RUN));
	my_map.insert(make_pair("GRABBER_CONTROLLED",  GRABBER_CONTROLLED));
// 		my_map.insert(make_pair("GRABBER_CONTROLLED_SYNCRON", GRABBER_CONTROLLED_SYNCRON));
	my_map.insert(make_pair("ASYNC_SOFTWARE_TRIGGER" , ASYNC_SOFTWARE_TRIGGER));
	my_map.insert(make_pair("ASYNC_TRIGGER" , ASYNC_TRIGGER));
	
// 		my_map.insert(make_pair("FG_STROBPULSEDELAY", FG_STROBPULSEDELAY));// Strobe delay to the trigger in µs.
	my_map.insert(make_pair("FG_TWOCAMMODEL", FG_TWOCAMMODEL));// Returns the value, if the loaded camera applet is a dual (1) or a single applet (0).
	my_map.insert(make_pair("FG_HDSYNC", FG_HDSYNC));// Returns the HDSYNC value. 
	my_map.insert(make_pair("FG_GLOBAL_ACCESS", FG_GLOBAL_ACCESS));// Returns the value for the set plausibility access.
	 
        /*
         * Information on the board type: 
         *
         *  BINFO_BOARD_TYPE:
         *      0xa40 for microEnable IV-Base x1
         *      0xa41 for microEnable IV-Full x1
         *      0xa44 for microEnable IV-Full x4
         *
         *  BINFO_POCL:
         *      0 for microEnable IV-Base x1
         *      1 for microEnable IV-Base x1 PoCL 
         */
	my_map.insert(make_pair("FG_BOARD_INFORMATION", FG_BOARD_INFORMATION));
}

static PyMethodDef VideoCapture_methods[] = {
	 {"read", (PyCFunction)VideoCapture_fgread, METH_NOARGS,
	 "read a frame from ximea device, return a tuple containing a bool (true= success, false= fail) and a dictionnary with a ndarray and meta data."},
	 {"set", (PyCFunction)VideoCapture_set, METH_VARARGS,
	 "set the configuration parameter specified of a ximea device"},
	 {"get", (PyCFunction)VideoCapture_get, METH_VARARGS,
	 "get the configuration parameter specified of a ximea device"},
	 {"isOpened", (PyCFunction)VideoCapture_isOpened, METH_NOARGS,
	 "return true if the ximea device is opened, false otherwise."},
	 {"release", (PyCFunction)VideoCapture_release, METH_NOARGS,
	 "release the ximea device."},
         {"startAcq", (PyCFunction)VideoCapture_startAcquisition, METH_NOARGS,
	 "Start the acquisition of a camera device"},
         {"stopAcq", (PyCFunction)VideoCapture_stopAcquisition, METH_NOARGS,
	 "Stop the acquisition of a camera device."},
	 {"serialWrite", (PyCFunction)VideoCapture_serialSet, METH_VARARGS,
	 "Write data to the serial port of a camera device."},
	 //{"show", (PyCFunction)VideoCapture_Display, METH_VARARGS,
	 //"display image."},
    {NULL}
};

static PyTypeObject VideoCaptureType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "VideoCapture.VideoCapture",             /*tp_name*/
    sizeof(VideoCapture),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)VideoCapture_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "VideoCapture objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    VideoCapture_methods,             /* tp_methods */
    VideoCapture_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)VideoCapture_init,      /* tp_init */
    0,                         /* tp_alloc */
    VideoCapture_new,                 /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}
};


/**
 * \fn PyMODINIT_FUNC initclModule(void) 
 * \brief This function will create the clModule and export the references to the VideoCapture class and other exported variables.
 */
PyMODINIT_FUNC 
initclModule(void) 
{
    try{
        PyObject* m;
		PyObject *tmp, *d;
		import_array();
        if (PyType_Ready(&VideoCaptureType) < 0)
			cout << "unable to install ximea module" << endl;

        m = Py_InitModule3("clModule", module_methods,
                        "Python module to control devices throught cameraLink interface");

	    if (m == NULL)
			cout << ( "unable to install ximea module" ) << endl;
		#if defined WIN32 || defined _WIN32
		set_map_to_export();
		#endif
		d = PyModule_GetDict(m);
		map<string, int>::iterator p;
		for(p = my_map.begin(); p != my_map.end(); p++)
		{
			tmp = Py_BuildValue("i", p->second);
			PyDict_SetItemString(d, (char *)p->first.c_str(), tmp);
			Py_DECREF(tmp);
		}
	            
        Py_INCREF(&VideoCaptureType);
        PyModule_AddObject(m, "VideoCapture", (PyObject *)&VideoCaptureType);
    } 
    catch ( const std::exception & e ) 
    { 
        std::cerr << e.what(); 
    } 
}
#ifdef __cplusplus
}
#endif 
/** @} */ 
/** @} */
/** @} */