#include <Python.h>
#include <iostream>

using namespace std;

typedef struct {
    PyObject_HEAD
    char *name;
} Hello;

static PyObject* Hello_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hello *self;
    self = (Hello *)type->tp_alloc(type, 0);
    static char *kwlist[] = {(char*)"name", NULL};
    if (self != NULL) {
        if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist,
            &self->name)){
                return NULL;
        }
    }
    return (PyObject *)self;
}

static int Hello_init(Hello *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {(char*)"name", NULL};

    self->name = (char*)"Crappy";
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &self->name)){
            return 1;
    }
    return 0;
}

static void Hello_dealloc(Hello* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject*
Hello_get(Hello *self)
{
    return Py_BuildValue("s", self->name);
}

PyObject*
Hello_print(Hello *self)
{
    cout << "Hello " << self->name << endl;
    Py_RETURN_NONE;
}

static PyMethodDef Hello_methods[] = {
    {"say_hello", (PyCFunction)Hello_print, METH_VARARGS,
     "Say hello to somebody."},
    {"get_name", (PyCFunction)Hello_get, METH_NOARGS,
    "Return the name attribute."},
    {NULL}
};

static PyTypeObject helloType = {
    PyObject_HEAD_INIT(NULL)
    .tp_name = "crappy.tool.Hello",
    .tp_basicsize = sizeof(Hello),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) Hello_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Hello objects",
    .tp_methods = Hello_methods,
    .tp_init = (initproc) Hello_init,
    .tp_new = Hello_new,
};

static struct PyModuleDef helloClassModule = {
    PyModuleDef_HEAD_INIT,
    "helloClassModule",
    NULL,
    -1,
    Hello_methods
};

PyMODINIT_FUNC
PyInit_helloClassModule(void)
{
    PyObject* m;
    PyType_Ready(&helloType);

    m = PyModule_Create(&helloClassModule);

    Py_INCREF(&helloType);
    PyModule_AddObject(m, "Hello", (PyObject *) &helloType);
    return m;
}
