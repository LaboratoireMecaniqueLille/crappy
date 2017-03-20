#include "CameraLink.h"
#include "export.h"
#include <numpy/arrayobject.h>
#include "structmember.h"

// =========== Methods for VideoCapture object ==============
// The constructor
static int VideoCapture_init(VideoCapture *self)
{
  //cout << "Calling init!" << endl;
  if(self->camptr != NULL)
    self->camptr->close();
  self->camptr = new CaptureCAM_CL();
  if(self->camptr != NULL) return 0;
  else return -1;
}

// And destructor
static void VideoCapture_destructor(PyObject *self)
{
  // cout << "Calling destructor!" << endl;
  ((VideoCapture*)self)->camptr->close();
  delete ((VideoCapture*)self)->camptr;
  cout << "cl Destructor done!" << endl;
}

PyObject* VideoCapture_open(VideoCapture* self, PyObject *args)
{
  int format = FG_GRAY16;
  if(!PyArg_ParseTuple(args,"is|i",&self->device,&self->camType,&format))
    return NULL;
  if(self->camptr->isopened) self->camptr->close();
  self->camptr->open(self->device,self->camType,format);
  return VideoCapture_isOpened(self);
}

PyObject* VideoCapture_isOpened(VideoCapture *self)
{
  if(self->camptr->isopened) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject* VideoCapture_release(VideoCapture *self)
{
  if(self->camptr->isopened)
    self->camptr->close();
  Py_RETURN_NONE;
}

PyObject* VideoCapture_read(VideoCapture *self)
{
  if(!self->camptr->grabFrame()) return Py_BuildValue("(OO)",Py_False,Py_None);
  return Py_BuildValue("(OO)",Py_True,VideoCapture_get_array(self));
}

PyObject* VideoCapture_get_array(VideoCapture *self)
{
  switch(self->camptr->format)
  {
  case FG_BINARY:
  case FG_GRAY:
    {
      npy_intp dims[2] = {self->camptr->height,self->camptr->width};
      return PyArray_SimpleNewFromData(2,dims,NPY_UINT8,self->camptr->ImgPtr);
    }
  case FG_GRAY16:
    {
      npy_intp dims[2] = {self->camptr->height,self->camptr->width};
      return PyArray_SimpleNewFromData(2,dims,NPY_UINT16,self->camptr->ImgPtr);
    }
  case FG_COL24:
    {
      cout << "24 bits!" << endl;
      npy_intp dims[3] = {self->camptr->height,self->camptr->width,3};
      return PyArray_SimpleNewFromData(3,dims,NPY_UINT8,self->camptr->ImgPtr);
    }
  default:
    {
      cout << "VideoCapture.get_array(): Unsupported data type!" << endl;
      Py_RETURN_NONE;
    }
  }
}

PyObject* VideoCapture_set(VideoCapture *self, PyObject *args)
{
	int propId;
	int value;
	if (!PyArg_ParseTuple(args, "ii", &propId, &value)) return NULL;
  if(self->camptr->setProperty(propId, value)) Py_RETURN_TRUE;
	else Py_RETURN_FALSE;
}

PyObject* VideoCapture_get(VideoCapture *self, PyObject *args)
{
	int propId;
	if (!PyArg_ParseTuple(args, "i", &propId)) return NULL;
  return Py_BuildValue("i", self->camptr->getProperty(propId));
}

PyObject* VideoCapture_startAcq(VideoCapture *self, PyObject *args)
{
  if(self->camptr->isacquiring) Py_RETURN_FALSE;
  if(!PyArg_ParseTuple(args,"|iii",
        &self->camptr->width,&self->camptr->height,&self->camptr->format))
    return NULL;
  switch(self->camptr->format)
  {
    case FG_GRAY:
      self->camptr->bytesPerPixel = 1;
      break;
    case FG_GRAY16:
      self->camptr->bytesPerPixel = 2;
      break;
    case FG_COL24:
      self->camptr->bytesPerPixel = 3;
      break;
  }
  if(!self->camptr->startAcquire())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* VideoCapture_stopAcq(VideoCapture *self)
{
  if(!self->camptr->isacquiring) Py_RETURN_FALSE;
  if(!self->camptr->stop())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* VideoCapture_serial_write(VideoCapture *self,PyObject *args)
{
  char *buffer;
  if(!PyArg_ParseTuple(args,"s",&buffer)) return NULL;
  return Py_BuildValue("s",self->camptr->serialWrite(buffer));
}

PyObject* VideoCapture_load_config(VideoCapture *self, PyObject *args)
{
  char *buffer;
  if(!PyArg_ParseTuple(args,"s",&buffer)) return NULL;
  self->camptr->loadConfig(buffer);
  Py_RETURN_NONE;
}


// Creating the PyDefMethod table grouping all the VideoCapture methods
static PyMethodDef VideoCapture_methods[] = {
  {"open",(PyCFunction)VideoCapture_open, METH_VARARGS,
    "Opens the frame grabber with default, needs the fg type as second argument"},
  {"read",(PyCFunction)VideoCapture_read, METH_NOARGS,
    "Read a frame from the camera, return a tuple with a bool set to True if"
    " reading was successful, False otherwise and the frame in second"},
  {"set",(PyCFunction)VideoCapture_set, METH_VARARGS,
    "Allows to set a parameter value for the frame grabber"},
  {"get",(PyCFunction)VideoCapture_get, METH_VARARGS,
    "Allows to get a parameter value from the frame grabber"},
  {"isOpened",(PyCFunction)VideoCapture_isOpened, METH_NOARGS,
    "Returns True if the device is open, False otherwise"},
  {"release",(PyCFunction)VideoCapture_release, METH_NOARGS,
    "Release the frame grabber, closing the device"},
  {"startAcq",(PyCFunction)VideoCapture_startAcq, METH_VARARGS,
    "Starts the acquisition, necessary to start capturing"},
  {"stopAcq",(PyCFunction)VideoCapture_stopAcq, METH_NOARGS,
    "Stops the acquisition, necessary to change settings"},
  {"serialWrite",(PyCFunction)VideoCapture_serial_write, METH_VARARGS,
    "To send a serial command to the camera through the cameralink"},
  {"loadFile",(PyCFunction)VideoCapture_load_config, METH_VARARGS,
    "To configure the frame grabber using a config file"},
  {NULL} // Sentinel
};

// The VideoCapture members
static PyMemberDef VideoCapture_members[] = {
  {NULL}
};

// Creating the type of the VideoCapture object
static PyTypeObject CLObjectType = {
	PyObject_HEAD_INIT(NULL)
	0,				/* ob_size        */
	"clModule.VideoCapture",		/* tp_name        */
	sizeof(VideoCapture),		/* tp_basicsize   */
	0,				/* tp_itemsize    */
	VideoCapture_destructor,				/* tp_dealloc     */
	0,				/* tp_print       */
	0,				/* tp_getattr     */
	0,				/* tp_setattr     */
	0,				/* tp_compare     */
	0,				/* tp_repr        */
	0,				/* tp_as_number   */
	0,				/* tp_as_sequence */
	0,				/* tp_as_mapping  */
	0,				/* tp_hash        */
	0,				/* tp_call        */
	0,				/* tp_str         */
	0,				/* tp_getattro    */
	0,				/* tp_setattro    */
	0,				/* tp_as_buffer   */
	Py_TPFLAGS_DEFAULT,		/* tp_flags       */
	"Object that represents a CL camera",	/* tp_doc         */
	0,				/* tp_traverse       */
	0,				/* tp_clear          */
	0,				/* tp_richcompare    */
	0,				/* tp_weaklistoffset */
	0,				/* tp_iter           */
	0,				/* tp_iternext       */
	VideoCapture_methods,	     		/* tp_methods        */
	VideoCapture_members,			/* tp_members        */
	0,				/* tp_getset         */
	0,				/* tp_base           */
	0,				/* tp_dict           */
	0,				/* tp_descr_get      */
	0,				/* tp_descr_set      */
	0,				/* tp_dictoffset     */
	(initproc)VideoCapture_init,		/* tp_init           */
  0,        /* tp_alloc */
  PyType_GenericNew        /* tp_new   */
};

// The module itself has no method
static PyMethodDef clModuleMethods[] = {
  {NULL} // Sentinel
};

// Initializing the module
PyMODINIT_FUNC initclModule(void)
{
  PyObject *m;
  if(!PyType_Ready(&CLObjectType) < 0)return;
  m = Py_InitModule3("clModule", clModuleMethods,
      "Module for cameralink interfaces");
  if(m == NULL){cout << "Unable to load clModule" << endl;return;}
  map<string, int>::iterator p;
  for(p=my_map.begin(); p != my_map.end();p++)
    PyModule_AddIntConstant(m,(char*)p->first.c_str(),p->second);


  Py_INCREF(&CLObjectType);
  // Add the python object to the module
  PyModule_AddObject(m,"VideoCapture",(PyObject*)&CLObjectType);
  import_array();
}
