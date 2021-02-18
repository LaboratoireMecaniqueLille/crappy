=======================================================
How to use C++ designed hardware drivers with Crappy ?
=======================================================

There is lot of hardware only available for C++ or C platforms, but it can be easily bound with Python. Here we will give you an example on how to bind C language to make it object oriented (as we use to in Python).

how to bind C/C++ with Crappy?
------------------------------

This gives a complete example on how to bind C and C++ language with Python and add it to the Crappy package. Linux and Windows are both used for building.

.. warning::
   This is not a C++ tutorial, some notion are used here, please refer to the tutorials
   below if you are not a C or C++ developer. /n C++ tutorials:

      - openclassrooms (fr)
      - cplusplus (en)

   C tutorials:

      - openclassroom (fr)
      - cprogramming (en)

Under Linux, you must install the python-dev package to ensure that you can use the Python.h library in the C or C++ code::

   sudo apt-get install python-dev

Under Windows, there is no python-dev package, but the python installer for windows will install a subdirectory in the python dir directory:

   C:\Python\include which contains the Python.h.

First example
--------------

The C++ code (hello.cpp)
+++++++++++++++++++++++++

::

   // Python header contains all the functions definitions to handle python object in C/C++.
   #include <Python.h>
   // Header that defines the standard input/output stream objects.
   #include <iostream>
   // define the namespace to use.
   using namespace std;
   // The functions bound with python have to return a PyObject understandable in Python.
   static PyObject* hello(PyObject* self, PyObject* args){
       const char* name;
       // it parse the args argument and look for a string
       // and set the name var with the parsed value.
       // if it fails, PyArg_ParseTuple return False, True otherwise.
       // returning NULL directly allows to raise an exception in Python.
       if(!PyArg_ParseTuple(args, "s", &name))
           return NULL;
       cout << "Hello " << name << endl;
       // This should be a void function, so we return the Python None.
       Py_RETURN_NONE;
   }
   // definition of a PyMethodDef object (defined in the python library)
   // contains the functions definition to be bound with python.
   static PyMethodDef HelloMethods[] =
   {
       {"hello", hello, METH_VARARGS, "Say hello to somebody."},
       {NULL, NULL, 0, NULL}
   };
   PyMODINIT_FUNC inithelloModule(void){
       (void) Py_InitModule("helloModule", HelloMethods);
   }

Each functions to bind have to return a PyObject pointer. Then, if a function needs to get arguments, there a passed to args, and first parsed in the function with PyArgs_ParseTuple. If the argument parsed is not a char it return NULL, returning a NULL alows to directly raise a python error. Here there is no need to return an object, so we return None (equivalent of C++ NULL).

Then, to bind the hello function we need to create a PyMethodDef which contains the function definition::

   {"hello", hello, METH_VARARGS, "Say hello to somebody."}

The first element will be the name of the function in python. The second element is the function to bind. The third element is METH_VARARGS if the function get arguments, or METH_NOARGS otherwise. The last element correspond to a description of the function, to appear in the function help.

Adding the binding to Crappy
+++++++++++++++++++++++++++++

To use the hello method defined in hello.cpp, we need to compile our project. This is automatically supported by the distutil package used to create the Crappy package.

We have to use the Extension module in distulil.core.

Example::

   helloModule = Extension('technical.helloModule',
                        sources=['sources/hello/hello.cpp'],
                        extra_compile_args=["-l", "python2.7"])
   extentions.append(helloModule)
   Extension take several argument, the first one is the full name of the extension, including any
   packages. Not a filename or pathname, but Python dotted name. Here we want to put the extension in
   technical, to import our module as crappy.technical.helloModule, so the extension name is
   'technical.helloModule'.

.. Note::

   Here, we called the extension helloModule, so the init method defined must be defined like follow::

      PyMODINIT_FUNC inithelloModule(void){
          (void) Py_InitModule("helloModule", HelloMethods);
      }

   the name of the function must be: init+[the name of your extension]: inithelloModule.
   Py_initModule must initialize a module with the same name "hellModule".

Extensions is just a list containing all the extensions to build, so we must add the helloModule to it.

Finally, we import our module in technical/__init__.py.

Now we can build our module with::

   sudo python setup.py install

The module helloModule.so will end up in /usr/local/lib/python2.7/dist-packages/crappy2-X.X.X-py2.7-linux-x86_64.egg/crappy2/technical and a helloModule.py file will be created to allow the import of the module::

   def __bootstrap__():
       global __bootstrap__, __loader__, __file__
       import sys, pkg_resources, imp
       __file__ = pkg_resources.resource_filename(__name__, 'helloModule.so')
       __loader__ = None; del __bootstrap__, __loader__
       imp.load_dynamic(__name__,__file__)
   __bootstrap__()

So we can now simply use our module::

   In [1]: import crappy2
   In [2]: crappy2.technical.helloModule.hello("Crappy")
   Hello Crappy

A more oriented object module
------------------------------

Let try to define a class that is similar to the following python class::

   class Hello:

       def __init__(self, name="Crappy"):
           self.name = name

       def say_hello(self):
           print 'hello ', self.name

       def get_name(self):
           return self.name

we first need to define the functions to construct our future class:

   - a new method
   - a constructor
   - a destructor And a structure which will contain the class attributes.

Here, the struct contains two elements. The first, PyObject_HEAD must be always defined, it represent the type of object. The second element represent our attribute 'name'.::

   // define a struct to build our Python module, this is similar to the dict of a Python class.
   typedef struct {
       PyObject_HEAD
       char *name;
   } Hello;

The new method parse the arguments and keywords arguments, to initialize the structure defined before, which will be passed as first argument for each method (similar to the python self).::

   // This function will be called at the creation of our Python class, it allocates memory, parse the
   arguments and return
   // the self struct.
   static PyObject *Hello_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
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

The constructor parses the arguments and keywords arguments. The "name" argument is optional: "\|s" string or nothing; name is set by default to "Crappy".::

   static int Hello_init(Hello *self, PyObject *args, PyObject *kwds)
   {
       static char *kwlist[] = {"name", NULL};
       self->name = "Crappy";
       if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &self->name)){
               return NULL;
       }
       return 0;
   }
   static void Hello_dealloc(Hello* self)
   {
       self->ob_type->tp_free((PyObject*)self);
   }

We then define our two method like before:

.. Note::

   To return a value, we need to use the Py_BuildValue function, to convert C++ type to python type: In
   this way, we directly get a understandable python object.::

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

To define a class which can be bound with Python, we need to define the structure of it, with a PyTypeObject. We have to define:

   - which function is the constructor
   - which one is the destructor, the new method...
   - what is the name of the class
   - its size
   - its methods

::

   static PyMethodDef Hello_methods[] = {
           {"say_hello", (PyCFunction)Hello_print, METH_NOARGS,
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
       0,                     /* tp_traverse */
       0,                     /* tp_clear */
       0,                     /* tp_richcompare */
       0,                     /* tp_weaklistoffset */
       0,                     /* tp_iter */
       0,                     /* tp_iternext */
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

Finally, as we did on the first example, the init method as to be defined::

   Py_InitModule3 create the module and return its instance (here empty).
   We can add our created objects, here helloType which defined our class.

.. Note::

   When returning an object, it returns a reference to it, each object has a reference counter this is
   made automatically for memory management issue, to know how many different places there are that have
   a reference to an object. When an object's reference count becomes 0, the object is automatically
   deallocated. This has to be made by yourself when dealing with C-C++/Python bindings. (With
   Py_INCREF, Py_DECREF). Please see Python C-api documentation for more details.

::

   PyMODINIT_FUNC
   inithelloModule(void)
   {
       try{
           PyObject* m;
           if (PyType_Ready(&helloType) < 0)
               cout << "unable to install ximea module" << endl;
           m = Py_InitModule3("helloModule", module_methods,
                              "hello C++ module.");
           Py_INCREF(&helloType);
           PyModule_AddObject(m, "Hello", (PyObject *)&helloType);
       }
       catch ( const std::exception & e )
       {
           std::cerr << e.what();
       }
   }

Example::

   In [2]: hello = crappy2.technical.helloModule.Hello("world")

   In [3]: hello.
   hello.get_name   hello.say_hello

   In [3]: hello.get_name()
   Out[3]: 'world'

   In [4]: hello.say_hello()
   Hello world

   In [5]: hello = crappy2.technical.helloModule.Hello()

   In [6]: hello.say_hello()
   Hello Crappy

   In [7]: hello.get_name()
   Out[7]: 'Crappy'
