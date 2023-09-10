#ifndef NIFGEN_H
#define NIFGEN_H
#include "Python.h"

#ifndef WIN32
#ifdef _WIN32
#define WIN32 _WIN32
#endif
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#include "niFgen.h"
#include <stdio.h>
#include <iostream>
#include <memory.h>
// #include <string>
#include <string>
#include <typeinfo>
#include <stdlib.h>
#include <datetime.h>
#include "structmember.h"
#include <utility>


using namespace std;

#ifdef __cplusplus
extern "C" {
#endif 


typedef struct {
    PyObject_HEAD
    char* device;
    ViReal64 Frequency, Amplitude, StartPhase, DCOffset;
    ViSession vi;
    ViInt32 wfmType;
} pyFgen;

PyObject* pyFgen_open(pyFgen *self);
PyObject* pyFgen_start(pyFgen *self);
PyObject* pyFgen_Configure(pyFgen *self, PyObject *args, PyObject *kwargs);
PyObject* pyFgen_stop(pyFgen *self);
PyObject* pyFgen_release(pyFgen *self);
#ifdef __cplusplus
}
#endif
#endif