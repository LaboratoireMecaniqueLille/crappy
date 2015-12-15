#include "ximea.h"
#include "export.h"
#include <numpy/arrayobject.h>
#include <datetime.h>
#include "structmember.h"
CaptureCAM_XIMEA* capt;

PyObject*
VideoCapture_open(int device)
{
    if (VideoCapture_isOpened() == Py_True) {
		VideoCapture_release();
	}
	capt->open(device);
    return VideoCapture_isOpened();
}

PyObject*
VideoCapture_isOpened()
{
    if(capt->isopened) {
		return Py_True;
	}else{
		return Py_False;
	}
}

PyObject*
VideoCapture_release()
{
    capt->close();
	return Py_None;
}


bool VideoCapture_grab()
{
    return capt->grabFrame();
}


PyObject* VideoCapture_retrieve(VideoCapture *self)
{
		char *array_buffer;
		short n;
		switch(capt->image.frm)
		{
		case XI_MONO8: {
			const int ndim = 2;
			npy_intp nd[2] = {capt->width, capt->height};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height);
			break;}
		case XI_MONO16:{ 
			const int ndim = 3;
			npy_intp nd[3] = {capt->width, capt->height, sizeof(n)};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT16);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height*sizeof(n));
			break;}
		case XI_RGB24       : {
			const int ndim = 3;
			npy_intp nd[3] = {capt->width, capt->height, 3};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height*3);
			break;}
		case XI_RGB32       : {
			const int ndim = 4;
			npy_intp nd[3] = {capt->width, capt->height, 4};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height*4);
			break;}
		case XI_RGB_PLANAR  : {
			const int ndim = 3;
			npy_intp nd[3] = {capt->width, capt->height, 3};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height*3);
			break;}
		case XI_RAW8        : {
			const int ndim = 2;
			npy_intp nd[2] = {capt->width, capt->height};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT8);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height);
			break;}
		case XI_RAW16       : {
			const int ndim = 3;
			npy_intp nd[3] = {capt->width, capt->height, sizeof(n)};
			self->myarray = PyArray_SimpleNew(ndim, nd, NPY_UINT16);
			array_buffer = (char *)PyArray_DATA((PyArrayObject *)self->myarray);
			memcpy(array_buffer, capt->image.bp, capt->width*capt->height*sizeof(n));
			break;}
		default : 
			return Py_None;
		}
		capt-> resetCvImage();
		
		PyObject *myDict = PyDict_New();
		myDict = VideoCapture_getMeta();
		PyDict_SetItemString(myDict, "data", self->myarray);
    return myDict;
}

PyObject*
VideoCapture_getMeta()
{
	PyDateTime_IMPORT;
	PyObject *myDict = PyDict_New();
// 	PyDict_SetItemString(myDict, "width", Py_BuildValue("I", capt->image.width));
// 	PyDict_SetItemString(myDict, "height", Py_BuildValue("I",capt->image.height));
// 	PyDict_SetItemString(myDict, "bp_size", Py_BuildValue("I",capt->image.bp_size));
// 	PyDict_SetItemString(myDict, "size", Py_BuildValue("I",capt->image.size));
// 	PyDict_SetItemString(myDict, "GPI_level", Py_BuildValue("I",capt->image.GPI_level));
// 	PyDict_SetItemString(myDict, "black_level", Py_BuildValue("I",capt->image.black_level));
// 	PyDict_SetItemString(myDict, "padding_x", Py_BuildValue("I",capt->image.padding_x));
	PyDict_SetItemString(myDict, "nframe", Py_BuildValue("I",capt->image.nframe));
	PyDict_SetItemString(myDict, "AbsoluteOffsetX", Py_BuildValue("I",capt->image.AbsoluteOffsetX));
	PyDict_SetItemString(myDict, "AbsoluteOffsetY", Py_BuildValue("I",capt->image.AbsoluteOffsetY));
	
	PyObject *floatObj = PyFloat_FromDouble(capt->image.tsSec);
	PyObject *timeTuple = Py_BuildValue("(O)", floatObj);  
	PyObject *dateTime = PyDateTime_FromTimestamp(timeTuple);
	PyDict_SetItemString(myDict, "tsSec", dateTime);
	
	PyObject *floatObj1 = PyFloat_FromDouble(capt->image.tsUSec);
	PyObject *timeTuple1 = Py_BuildValue("(O)", floatObj1);  
	PyObject *dateTime1 = PyDateTime_FromTimestamp(timeTuple1);
	PyDict_SetItemString(myDict, "tsUSec", dateTime1);
	
	return myDict;
}

PyObject*
VideoCapture_xiread(VideoCapture *self)
{
	PyObject *rslt = PyTuple_New(2);
	if(!VideoCapture_grab()){
		PyTuple_SetItem(rslt, 0, Py_False);
		PyTuple_SetItem(rslt, 1, Py_None);
    }else{
		PyTuple_SetItem(rslt, 0, Py_True);
		PyTuple_SetItem(rslt, 1, VideoCapture_retrieve(self));
    }
    return rslt;
    
}

PyObject*
VideoCapture_set(VideoCapture *self, PyObject *args)
{
	int propId;
	double value;
	if (!PyArg_ParseTuple(args, "id", &propId, &value))
		exit(0);
    if(capt->setProperty(propId, value)) return Py_True;
	else return Py_False;
}

PyObject*
VideoCapture_get(VideoCapture *self, PyObject *args)
{
	int propId;
	if (!PyArg_ParseTuple(args, "i", &propId))
		exit(0);
    if(capt->getProperty(propId)) return Py_True;
	else return Py_False;
}

static void
VideoCapture_dealloc(VideoCapture* self)
{
    Py_XDECREF(self->myarray);
	VideoCapture_release();
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
VideoCapture_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    VideoCapture *self;

    self = (VideoCapture *)type->tp_alloc(type, 0);
    if (self != NULL) {
		if (!PyArg_ParseTuple(args, "i:call", &self->device)) {
			return NULL;
		}
    }

    return (PyObject *)self;
}

static int
VideoCapture_init(VideoCapture *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"device", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,  
                                      &self->device))
        return -1; 
	capt = new CaptureCAM_XIMEA();
	VideoCapture_open(self->device);
    return 0;
}


static PyMemberDef VideoCapture_members[] = {
    {NULL} 
};


static PyMethodDef VideoCapture_methods[] = {
    {"read", (PyCFunction)VideoCapture_xiread, METH_NOARGS,
	 "read a frame from ximea device, return a tuple containing a bool (true= success, false= fail) and a dictionnary with an ndarray and meta."},
	 {"set", (PyCFunction)VideoCapture_set, METH_VARARGS,
	 "set the configuration parameter specified of a ximea device"},
	 {"get", (PyCFunction)VideoCapture_get, METH_VARARGS,
	 "get the configuration parameter specified of a ximea device"},
	 {"isOpened", (PyCFunction)VideoCapture_isOpened, METH_NOARGS,
	 "return true if the ximea device is opened, false otherwise."},
	 {"release", (PyCFunction)VideoCapture_release, METH_NOARGS,
	 "release the ximea device."},
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

PyMODINIT_FUNC
initximeaModule(void) 
{
    PyObject* m;
	PyObject *tmp, *d;
	import_array();
    if (PyType_Ready(&VideoCaptureType) < 0)
        return;

    m = Py_InitModule3("ximeaModule", module_methods,
                       "Example module that creates an extension type.");

    if (m == NULL)
      return;
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