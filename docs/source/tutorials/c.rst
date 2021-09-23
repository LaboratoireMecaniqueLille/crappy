====================================
How to use C or C++ code in Crappy ?
====================================

If you want to use hardware that's only available on C or C++ platforms, or if
you need to run computationally intensive code in an efficient way, it is quite
easy to bind C/C++ code to Crappy.

1. Using an existing library
----------------------------

If you want to integrate a library from an external source and you already have
the corresponding file (``.so``, ``.dll``, etc.), different solutions are
available. We're not going to detail them here, as we could not provide a
cross-platform compatible example. You can however look into the `pyspcm
<https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/crappy/tool/
pyspcm.py>`_ and `comedi_bind <https://github.com/LaboratoireMecaniqueLille/
crappy/blob/master/crappy/tool/comedi_bind.py>`_ tools for an example using the
:mod:`ctypes` library and ``.so`` files. Alternative solutions include Cython,
:mod:`cffi`, PyBind11, and many others.

2. Writing your own library
---------------------------

2.1. Prerequisites
++++++++++++++++++

Here is detailed a complete example on how to bind C and C++ language with
Python and add it to the Crappy package. Linux and Windows are both used for
building.

.. Note::
  This is not a C++ tutorial, a few programming notions are used here, please
  refer to the tutorials below if you are not a C or C++ developer.

  C++ tutorials :

  - `openclassrooms C++ <https://openclassrooms.com/fr/courses/1894236-
    programmez-avec-le-langage-c>`_ (fr)
  - `cplusplus <https://www.cplusplus.com/doc/tutorial/>`_ (en)

  C tutorials :

  - `openclassrooms C <https://openclassrooms.com/fr/courses/19980-apprenez-a-
    programmer-en-c>`_ (fr)
  - `cprogramming <https://www.cprogramming.com/tutorial/c-tutorial.html?
    inl=hp>`_ (en)

In Linux, you must install the ``python-dev`` package to ensure that you can use
the ``Python.h`` library in the C or C++ code :

.. code-block:: shell

  sudo apt install python-dev

In Windows, there is no ``python-dev`` package, but the Python installer for
Windows will install a subdirectory in the Python directory which contains the
``Python.h`` (on our Windows machine, this folder is located in
``C:\Users\<username>\AppData\Local\Programs\Python\``).

.. Important::
   In this tutorial we'll assume that you're using Crappy from a ``setup``
   install (see the :ref:`Installation` section for details). Using C++ modules
   is otherwise not possible using only a ``pip`` install, as we made the choice
   not to distribute compiled code.

2.2. First example: a simple function
+++++++++++++++++++++++++++++++++++++

2.2.1. Writing the C++ code
"""""""""""""""""""""""""""

First we need to start with including the ``Python.h`` library, and of course
any other library we want to use. Then we write our function, that must return a
``PyObject`` pointer. If the function takes arguments (that's the case in the
example), they should be passed to ``args``. The arguments then have to be
parsed using ``PyArgs_ParseTuple``, and we can finally write the main part of
the function. Do not forget to add ``Py_RETURN_NONE`` at the end if the function
doesn't return anything.

.. code-block:: c

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

Then, to bind the ``hello`` function we need to create a ``PyMethodDef`` which
contains the function definition. If several functions were to be defined, we
would list them all here. The first element will be the name of the function in
Python. The second element is the function to bind. The third element is
``METH_VARARGS`` if the function gets arguments, or ``METH_NOARGS`` otherwise.
The last element corresponds to a description of the function, that will appear
in the function help.

.. code-block:: c
   :emphasize-lines: 15-19

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

Once all the Python objects have been defined, we need to define the Python
module itself. This is done using ``PyModuleDef``. It has to be initialized
with ``PyModuleDef_HEAD_INIT``, then comes the module name, the docstring if
any, the size to allocate to the module, the methods of the module, anf finally
the slots.

.. code-block:: c
   :emphasize-lines: 21-27

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

The last step is to initialize the module, which is done using
``PyMODINIT_FUNC`` and ``PyModule_Create``. The C++ code is then ready to be
compiled !

.. code-block:: c
   :emphasize-lines: 29-32

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

2.2.2. Binding the code to Crappy
"""""""""""""""""""""""""""""""""

Once the C++ code is written, most of the work is done. We only need to modify
the ``setup.py`` and one of the ``__init__.py`` files. So let's start with the
``setup.py``. All we have to do is include our ``.cpp`` file as an extension,
which is achieved by writing :

.. code-block:: python

   helloModule = Extension('tool.helloModule',
                           sources=['sources/hello/hello.cpp'],
                           extra_compile_args=["-l", "python%s" % v],
                           language='c++')

   extensions.append(helloModule)

This should be put around line 30. The first argument indicates where to write
the binary file that will be generated, the second points to the location(s) of
the ``.cpp`` and/or ``.hpp`` files to use for the extension, and the two last
arguments should be left as is.

Then the module should be imported in the ``__init__.py`` file of the folder
where the compiled file will be written, so in our example in ``crappy/tool/``.
The import is similar to all the regular ones, i.e. in our example we should
write :

.. code-block:: python

   from .helloModule import hello

The last step is to reinstall Crappy, and that's it ! During install any error
or warning related to the compilation of the C files will be displayed. After
completing the install, there should be no notable change in the source folder.
If you go to the install folder (see :ref:`here <2. Permanently adding custom
blocks to Crappy>`), there should be a binary file in the ``tool`` folder as
well as a ``helloModule.py`` file. This file contains the following code :

.. code-block:: python

   def __bootstrap__():
       global __bootstrap__, __loader__, __file__
       import sys, pkg_resources, imp
       __file__ = pkg_resources.resource_filename(__name__, '<binary_file.xxx>')
       __loader__ = None; del __bootstrap__, __loader__
       imp.load_dynamic(__name__,__file__)
   __bootstrap__()

It's this file that actually allows the import in ``__init__.py`` to happen. So
the ``hello`` method is now part of Crappy and lives in ``crappy.tool.hello``.
For using it in a script or in a command line, we can simply write :

  >>> import crappy
  >>> crappy.tool.hello('world')
  Hello world

2.3. Second example: a simple class
+++++++++++++++++++++++++++++++++++

2.3.1 Writing the C++ code
""""""""""""""""""""""""""

Now let's try to build a more advanced Python object in C. We'll define a class
that is similar to this Python class :

.. code-block:: python

  class Hello:

      def __init__(self, name="Crappy"):
          self.name = name

      def say_hello(self):
          print 'hello ', self.name

      def get_name(self):
          return self.name

After including the necessary packages, we first need to define the functions to
construct our future class :

- a new method
- a constructor
- a destructor and a structure which will contain the class attributes.

Here, the ``struct`` contains two elements. The first, ``PyObject_HEAD`` must
always be defined, it represent the type of object. The second element
represents our attribute ``'name'``.

.. code-block:: c

   #include <Python.h>
   #include <iostream>

   using namespace std;

   typedef struct {
       PyObject_HEAD
       char *name;
   } Hello;

The next method parses the arguments and keywords arguments, to initialize the
structure defined before, which will be passed as first argument for each method
(similar to the Python ``self``).

.. code-block:: c
   :emphasize-lines: 11-23

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

The constructor parses the arguments and keywords arguments. The ``"name"``
argument is optional. Here's also the destructor.

.. code-block:: c
   :emphasize-lines: 25-39

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

We then define our two methods like previously. To return a value, we need to
use the ``Py_BuildValue`` function, to convert C++ type to python type: this
way, we directly get an understandable Python object.

.. code-block:: c
   :emphasize-lines: 41-52

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

Now just like in the previous example we need to list the different methods of
our module.

.. code-block:: c
   :emphasize-lines: 54-60

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

To define a class which can be bound with Python, we need to define its
structure, with a ``PyTypeObject``. We have to define:

- the constructor
- the destructor
- the new method
- the name of the class
- its size
- its methods
- etc.

.. code-block:: c
   :emphasize-lines: 62-73

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

Finally just like in the first example we have to define the module and to
initialize it. Here the syntax is a bit more complex but the idea remains the
same.

.. code-block:: c
   :emphasize-lines: 75-94

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

2.3.2 Binding the code to Crappy
""""""""""""""""""""""""""""""""

Now that the C++ code is ready, let's add it to the extensions in ``setup.py`` :

.. code-block:: python

   helloClassModule = Extension('tool.helloClassModule',
                                sources=['sources/hello/hello_class.cpp'],
                                extra_compile_args=["-l", "python%s" % v],
                                language='c++')

   extensions.append(helloClassModule)

We also need to import it from ``__init__.py`` in ``crappy/tool/`` :

.. code-block:: python

   from .helloClassModule import Hello

After reinstalling Crappy, we can now use our class very simply :

  >>> import crappy
  >>> default = crappy.tool.Hello()
  >>> default.get_name()
  'Crappy'
  >>> default.say_hello()
  Hello Crappy
  >>> with_arg = crappy.tool.Hello('Bob')
  >>> with_arg.get_name()
  'bob'
  >>> with_arg.say_hello()
  Hello bob
