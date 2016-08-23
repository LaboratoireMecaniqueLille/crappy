#include <Python.h>
#include <iostream>

using namespace std;

static PyObject* hello(PyObject* self, PyObject* args){
    const char* name;
    if(!PyArg_ParseTuple(args, "s", &name))
        return NULL;
    cout << "Hello " << name << endl;

    Py_RETURN_NONE;
}

static PyMethodDef HelloMethods[] =
{
    {"hello", hello, METH_VARARGS, "Say hello to somebody."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inithelloModule(void){
    (void) Py_InitModule("helloModule", HelloMethods);
}