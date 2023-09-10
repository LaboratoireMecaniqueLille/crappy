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

static struct PyModuleDef helloModule = {
    PyModuleDef_HEAD_INIT,
    "helloModule",
    NULL,
    -1,
    HelloMethods
};

PyMODINIT_FUNC PyInit_helloModule(void)
{
    return PyModule_Create(&helloModule);
}