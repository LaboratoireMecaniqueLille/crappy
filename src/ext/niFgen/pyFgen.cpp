#include "pyFgen.h"
#include <map>

map< string, long > export_map;
const ViChar * ChannelName = "0";
#ifdef __cplusplus
extern "C" {
#endif

PyObject*
pyFgen_Configure(pyFgen *self, PyObject *args, PyObject *kwargs)
{
   long outputMode, waveType;
   float amp, offset, freq, phase;

   waveType = NIFGEN_VAL_WFM_SINE;
   freq = 1e+6;
   amp = 1.0;
   phase = 0.0;
   offset = 0.0;

   waveType = NIFGEN_VAL_WFM_SINE;
   outputMode = NIFGEN_VAL_OUTPUT_FUNC;
   static char *kwlist[] = {
      "outputMode", /* bytes object. */
      "wfmType",
      "Amplitude",
      "DCOffset",
      "Frequency",
      "StartPhase",
      NULL
   };

   if (! PyArg_ParseTupleAndKeywords(args, kwargs, "|llffff",
                                      kwlist, &outputMode, &waveType, &amp, &offset, &freq, &phase)){
                     return NULL;
    }

   self->wfmType = waveType;
   self->Amplitude = amp;
   self->DCOffset = offset;
   self->Frequency = freq;
   self->StartPhase = phase;

   cout << "outputMode: " << outputMode << endl;
   cout << "waveType: " << waveType << endl;
   cout << "amp: " << amp << endl;
   cout << "offset: " << offset << endl;
   cout << "freq: " << freq << endl;
   cout << "phase: " << phase << endl << endl;

   cout << "outputMode: "  << outputMode << endl;
   cout << "wfmType: " << self->wfmType << endl;
   cout << "Amplitude: " << self->Amplitude << endl;
   cout << "DCOffset: " << self->DCOffset << endl;
   cout << "Frequency: " << self->Frequency << endl;
   cout << "StartPhase: " << self->StartPhase << endl;
   /*- Configure output for standard function mode -------------------------*/
   ViStatus error = VI_SUCCESS;
   checkErr(niFgen_ConfigureOutputMode(self->vi,outputMode)); // NIFGEN_VAL_OUTPUT_FUNC));

   /*- Configure the standard function to generate ------------------------- */
   checkErr(niFgen_ConfigureStandardWaveform(self->vi, ChannelName, self->wfmType,
                                             self->Amplitude, self->DCOffset, self->Frequency,
                                             self->StartPhase));
   Error:
      /*- Process any errors ---------------------------------------------------*/
      if(error != VI_SUCCESS) {
         ViChar errMsg[256];
         niFgen_ErrorHandler(self->vi, error, errMsg);
         printf("Error %x: %s\n", error, errMsg);
      }
      Py_INCREF(Py_None);
      return Py_None;
}


PyObject*
pyFgen_start(pyFgen *self)
{
   try{
      ViStatus error = VI_SUCCESS;
      checkErr(niFgen_ConfigureOutputEnabled(self->vi, ChannelName, VI_TRUE));
      checkErr(niFgen_InitiateGeneration(self->vi));
         /*- Convert strings to constants ----------------------------------------*/
      ViChar Type[256] = "";
      if      (self->wfmType = NIFGEN_VAL_WFM_SINE) strcpy(Type, "sine");
      else if (self->wfmType = NIFGEN_VAL_WFM_SQUARE) strcpy(Type, "square");
      else if (self->wfmType = NIFGEN_VAL_WFM_TRIANGLE) strcpy(Type, "triangle");
      else if (self->wfmType = NIFGEN_VAL_WFM_RAMP_UP) strcpy(Type, "up");
      else if (self->wfmType = NIFGEN_VAL_WFM_RAMP_DOWN) strcpy(Type, "down");
      else if (self->wfmType = NIFGEN_VAL_WFM_DC) strcpy(Type, "dc");
      else if (self->wfmType = NIFGEN_VAL_WFM_NOISE) strcpy(Type, "noise");
      else strcpy(Type, "None");

      printf("Generating a %s wave\n", Type);

      Error:
         /*- Process any errors ---------------------------------------------------*/
         if(error != VI_SUCCESS) {
            ViChar errMsg[256];
            niFgen_ErrorHandler(self->vi, error, errMsg);
            printf("Error %x: %s\n", error, errMsg);
         }
   } catch (...) {
       cout << "Unexpected exception." << endl;
   }
   Py_INCREF(Py_None);
   return Py_None;
}
PyObject*
pyFgen_open(pyFgen *self)
{
   self->vi=VI_NULL;
   ViChar Resource[256] = "";
   ViStatus error = VI_SUCCESS;
 
   // set default values
   strcpy(Resource, self->device);
   self->wfmType = NIFGEN_VAL_WFM_SINE;
   self->Frequency = 1e+6;
   self->Amplitude = 1.0;
   self->StartPhase = 0.0;
   self->DCOffset = 0.0;

   /*- Initialize the session ----------------------------------------------*/
   printf("Initializing %s\n", Resource);
   checkErr(niFgen_init(Resource, VI_TRUE, VI_TRUE, &self->vi));
   /*- Configure the active channels for the session -----------------------*/
   checkErr(niFgen_ConfigureChannels(self->vi, "0"));
   /*- Enable output and generate ------------------------------------------*/

Error:
   /*- Process any errors ---------------------------------------------------*/
   if(error != VI_SUCCESS) {
      ViChar errMsg[256];
      niFgen_ErrorHandler(self->vi, error, errMsg);
      printf("Error %x: %s\n", error, errMsg);
   }
   Py_INCREF(Py_None);
   return Py_None;
}


PyObject*
pyFgen_stop(pyFgen *self)
{
   if (self->vi) {
      niFgen_AbortGeneration(self->vi);
   }
   Py_INCREF(Py_None);
   return Py_None;
}

PyObject*
pyFgen_release(pyFgen *self)
{
   if (self->vi) {
      niFgen_AbortGeneration(self->vi);
      niFgen_close (self->vi);
   }
   Py_INCREF(Py_None);
   return Py_None;
}

static void
pyFgen_dealloc(pyFgen* self)
{
    pyFgen_release(self);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
pyFgen_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pyFgen *self;

    self = (pyFgen *)type->tp_alloc(type, 0);
    if (self != NULL) {
      if (!PyArg_ParseTuple(args, "s:call", &self->device)) {
         return NULL;
      }
    }

    return (PyObject *)self;
}

static int
pyFgen_init(pyFgen *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"device", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist,  
                                      &self->device))
        return -1; 
   pyFgen_open(self);
    return 0;
}


static PyMemberDef pyFgen_members[] = {
    {NULL} 
};

const char* conf_doc = "It configures the wave to generate, with keyword arguments: \n \
      - outputMode: \n \
                     - NIFGEN_VAL_OUTPUT_FUNC \n \
                     - NIFGEN_VAL_OUTPUT_ARB \n \
                     - NIFGEN_VAL_OUTPUT_SEQ \n \
                     - NIFGEN_VAL_OUTPUT_FREQ_LIST \n \
                     - NIFGEN_VAL_OUTPUT_SCRIPT \n \
      - wfmType: \n \
                     - NIFGEN_VAL_WFM_SINE \n \
                     - NIFGEN_VAL_WFM_SQUARE \n \
                     - NIFGEN_VAL_WFM_TRIANGLE \n \
                     - NIFGEN_VAL_WFM_RAMP_UP \n \
                     - NIFGEN_VAL_WFM_RAMP_DOWN \n \
                     - NIFGEN_VAL_WFM_DC \n \
                     - NIFGEN_VAL_WFM_NOISE \n \
      - Amplitude \n \
      - DCOffset \n \
      - Frequency \n \
      - StartPhase \n \
   ";
static PyMethodDef pyFgen_methods[] = {

   {"start", (PyCFunction)pyFgen_start, METH_NOARGS,
    "start method"},
   {"Configure", (PyCFunction)pyFgen_Configure, METH_VARARGS|METH_KEYWORDS, conf_doc },
   {"stop", (PyCFunction)pyFgen_stop, METH_NOARGS,
    "stop method"},
   {"release", (PyCFunction)pyFgen_release, METH_NOARGS,
    "release method"},
    {NULL, NULL}
};

static PyTypeObject pyFgenType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "waveGenerator",             /*tp_name*/
    sizeof(pyFgen),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)pyFgen_dealloc, /*tp_dealloc*/
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
    "pyFgen objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    pyFgen_methods,             /* tp_methods */
    pyFgen_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)pyFgen_init,      /* tp_init */
    0,                         /* tp_alloc */
    pyFgen_new,                 /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}
};

void set_map_to_export(){
   export_map.insert(make_pair("NIFGEN_VAL_WFM_SINE" , NIFGEN_VAL_WFM_SINE));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_SQUARE" , NIFGEN_VAL_WFM_SQUARE));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_TRIANGLE" , NIFGEN_VAL_WFM_TRIANGLE));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_RAMP_UP" , NIFGEN_VAL_WFM_RAMP_UP));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_RAMP_DOWN" , NIFGEN_VAL_WFM_RAMP_DOWN));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_DC" , NIFGEN_VAL_WFM_DC));
   export_map.insert(make_pair("NIFGEN_VAL_WFM_NOISE" , NIFGEN_VAL_WFM_NOISE));
   export_map.insert(make_pair("NIFGEN_VAL_OUTPUT_FUNC" , NIFGEN_VAL_OUTPUT_FUNC));
   export_map.insert(make_pair("NIFGEN_VAL_OUTPUT_ARB", NIFGEN_VAL_OUTPUT_ARB));
   export_map.insert(make_pair("NIFGEN_VAL_OUTPUT_SEQ", NIFGEN_VAL_OUTPUT_SEQ));
   export_map.insert(make_pair("NIFGEN_VAL_OUTPUT_FREQ_LIST", NIFGEN_VAL_OUTPUT_FREQ_LIST));
   export_map.insert(make_pair("NIFGEN_VAL_OUTPUT_SCRIPT", NIFGEN_VAL_OUTPUT_SCRIPT));
   export_map.insert(make_pair("VI_TRUE" , VI_TRUE));
   export_map.insert(make_pair("VI_SUCCESS", VI_SUCCESS));
   export_map.insert(make_pair("VI_NULL", VI_NULL));
}

PyMODINIT_FUNC
initpyFgenModule(void) 
{
   try{
      PyObject* m;
      PyObject *tmp, *d;
      if (PyType_Ready(&pyFgenType) < 0)
         cout << "unable to install Fgen module" << endl;

      m = Py_InitModule3("pyFgenModule", module_methods,
                          "Generating wave with National Instrument card compatible with FGEN Driver.");

      if (m == NULL)
         cout << ( "unable to install Fgen module" ) << endl;
      set_map_to_export();
      d = PyModule_GetDict(m);
      map<string, long>::iterator p;
      for(p = export_map.begin(); p != export_map.end(); p++)
      {
         tmp = Py_BuildValue("l", p->second);
         PyDict_SetItemString(d, (char *)p->first.c_str(), tmp);
         Py_DECREF(tmp);
      }
      Py_INCREF(&pyFgenType);
      PyModule_AddObject(m, "waveGenerator", (PyObject *)&pyFgenType);
    } 
    catch ( const std::exception & e ) 
    { 
        std::cerr << e.what(); 
    } 
}
#ifdef __cplusplus
}
#endif 