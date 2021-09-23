#include <Python.h>
#include <iostream>
using namespace std;

typedef struct {
    PyObject_HEAD
    char *name;
} Hello;



static PyObject *
Hello_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Hello *self;
    self = (Hello *)type->tp_alloc(type, 0);
    static char *kwlist[] = {"name", NULL};
    if (self != NULL) {
        if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &self->name)){
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
    self->ob_type->tp_free((PyObject*)self);
}

PyObject*
Hello_get(Hello *self)
{
//    PyObject *ret_val =
//    return Py_INCREF(ret_val);
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

static PyMethodDef module_methods[] = {
    {NULL}
};

static PyTypeObject helloType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "Hello",             /*tp_name*/
    sizeof(Hello),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Hello_dealloc, /*tp_dealloc*/
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
    "Hello objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Hello_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Hello_init,      /* tp_init */
    0,                         /* tp_alloc */
    Hello_new,                 /* tp_new */
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
