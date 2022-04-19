============
Installation
============

Requirements
------------

You can install Crappy on the three main OS : Windows, Mac and Linux. Note that
Crappy is developed and tested on the latest OS versions, and no particular
effort is made to ensure compatibility with previous ones. It is thus preferable
to install it on a recent and up-to-date OS version.

To install Crappy, you will need Python 3 (3.6 or higher) with the following
modules :

- `numpy <https://numpy.org/>`_ (1.19.0 or higher)

The following modules are not mandatory but will provide additional
functionalities (this list is not exhaustive) :

- `matplotlib <https://matplotlib.org/>`_ (1.5.3 or higher, for plotting graphs
  and displaying images)
- `opencv <https://opencv.org/>`_ (3.0 or higher, to perform image acquisition
  and analysis)
- `pyserial <https://pypi.org/project/pyserial/>`_ (To interface with serial
  sensors and actuators)
- `Tk <https://docs.python.org/3/library/tkinter.html>`_ (For the configuration
  interface of cameras)
- `scikit-image <https://scikit-image.org/>`_ (0.11 or higher)
- `xiApi <https://www.ximea.com/support/wiki/apis/xiapi>`_ (for Ximea cameras)
- `labjack <https://labjack.com/support/software/examples/ljm/python>`_ (for
  Labjack support)
- `Simple-ITK <https://simpleitk.org/>`_ (for faster image saving)
- `PyCUDA <https://documen.tician.de/pycuda/>`_ (for real-time correlation)
- `Comedi <https://www.comedi.org/>`_ driver (for Comedi acquisition boards,
  Linux only)
- `PyDaqmx <https://pythonhosted.org/PyDAQmx/>`_ (for National Instrument
  boards, Windows only)
- `niFgen <https://www.ni.com/fr-fr/support/downloads/drivers/
  download.ni-fgen.html>`_ (package from National Instrument, Windows only)
- `openDAQ <https://pypi.org/project/opendaq/>`_ (for opendaq boards)

.. note::
  Knowing which modules are needed for a given setup is easy. Just write the
  script and start it, if a module is missing Crappy will simply tell you !

A. For Linux users
------------------
These steps have been tested for Ubuntu 18.04 and 20.04 but should work on other
distributions as well, as long as Python 3.6 is installed.

**1.** First make sure that you have ``pip`` installed.

.. code-block:: shell-session

   sudo apt update
   sudo apt upgrade
   sudo apt install python3-pip

**2.** Then it's time to install Crappy ! There are three possible ways.

**2.a.** For **regular users**, install Crappy in a ``virtualenv`` (recommended) :

.. code-block:: shell-session

  workon myenv
  pip3 install crappy

If you're not familiar with virtual environments,
`here <https://virtualenv.pypa.io/en/latest/>`_'s more documentation.

**2.b.** For **regular users**, install Crappy on the system :

.. code-block:: shell-session

  pip3 install crappy

**2.c.** For **developers**, get the sources using ``git`` and use ``setup``
script :
::

  cd <path>
  git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
  cd crappy
  sudo python3 setup.py install

This installation can also be done in a virtual environment.
If you're not familiar with ``git``, documentation can be found
`there <https://git-scm.com/doc>`_.

.. important::
  For adding C/C++ modules to Crappy you **must** run a ``setup`` install, so
  you need to get Crappy from ``git``.

**3.** Finally, you can install additional packages to extend Crappy's functionalities.

**3.a.** In a ``virtualenv`` :

.. code-block:: shell-session

  workon myenv
  pip3 install <module>

**3.b.** Or system-wide :

.. code-block:: shell-session

  pip3 install <module>

**3.c** It is also possible to use the extras for installing additional
dependencies directly along with Crappy :

.. code-block:: shell-session

  pip3 install crappy[<extra>]

Currently, the available extras are ``SBC``, ``image``, ``hardware`` and ``main``.
They contain respectively modules for interfacing with single board computers,
for recording and displaying images and videos, for interfacing with hardware over
serial or USB, and ``main`` contains the three most used modules in Crappy
after the mandatory Numpy.

.. note::
  - Replace ``<module>`` by the name of the module you want to install.
  - Replace ``<path>`` by the path where you want Crappy to be located.
  - Replace ``<extra>`` by the name of the extra to install.

B. For Windows users
--------------------
These steps have been tested for Windows 8.1 and 10 but should work with other
versions as well. If you want to load C++ modules, make sure to use the x64
version of Python.

**1.** Install the dependencies :

.. code-block:: shell-session

  pip install <module>

This will works for most modules, but some may fail and need a wheel file built
for Windows. Once you've found a binary wheel, simply run :

.. code-block:: shell-session

  pip install <wheel_file.whl>

**2.** Also, you will need Visual C++ for Python 3.x (your version of python) if
you want to compile C++ modules.  If you want to use Ximea cameras, don't
forget to install XiAPI and add ``c:\XIMEA\API\x64`` to your path.

**3.** Then you can install Crappy.

If you're a **regular user** :

.. code-block:: shell-session

  pip install crappy

Or if you're a **developer** :

.. code-block:: shell-session

  cd <path>
  git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
  cd crappy
  setup.py install

If you're not familiar with ``git``, documentation can be found
`there <https://git-scm.com/doc>`_.

.. important::
  For adding C/C++ modules to Crappy you **must** run a ``setup`` install, which
  will be way more convenient if you get Crappy from ``git``.

.. note::
  - Replace ``<module>`` by the name of the module you want to install.
  - Replace ``<path>`` by the path where you want Crappy to be located.

C. For macOS users
------------------
These steps have been tested on macOS Sierra (10.12.6), but should work with
other versions as well.

**1.** Install the dependencies :

.. code-block:: shell-session

  pip3 install <module>

**2.** Then you can install Crappy.

If you're a **regular user** :

.. code-block:: shell-session

  pip3 install crappy

Or if you're a **developer** :

.. code-block:: shell-session

  cd <path>
  git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
  cd crappy
  setup.py install

If you're not familiar with ``git``, documentation can be found
`there <https://git-scm.com/doc>`_.

.. note::
  - Replace ``<module>`` by the name of the module you want to install.
  - Replace ``<path>`` by the path where you want Crappy to be located.

D. Troubleshooting
------------------

The imaging module is not natively included in Tk. Some user may have to install
it manually to use the camera configuration GUI.

For Ubuntu, you can do :

.. code-block:: shell-session

  sudo apt install python3-pil.imagetk

Also, you may face some issues with matplotlib backends not managing to open
multiple windows in some desktop environment. We set the default backend to
``TkAgg``, which works fine in most situations. If you encounter backend issues,
you can specify another backend for matplotlib in the grapher blocks :

.. code-block:: shell-session

  graph = crappy.bocks.Grapher(<args, kwargs>, backend='TkAgg')
