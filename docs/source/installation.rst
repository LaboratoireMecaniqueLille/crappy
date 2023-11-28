============
Installation
============

.. role:: shell(code)
  :language: shell-session
  :class: highlight

Requirements
------------

Crappy was successfully installed and tested on **Linux** (Ubuntu 18.04 and
higher), **Windows** (8 and higher) and **MacOS** (Sierra and higher). It was
also successfully installed on **Raspberry Pi** 3B+ and 4B. As a Python module,
Crappy can probably be installed on other systems able to run Python, but that
was not tested.

.. note::
  We develop Crappy on recent OS versions, and no particular effort is made to
  ensure compatibility with older OS versions.

Crappy requires **Python 3.6 or later**, as well as the following module :

- `numpy <https://numpy.org/>`_ (1.21.0 or higher)

The following modules are not mandatory but will provide additional
functionalities (this list is not exhaustive) :

- `matplotlib <https://matplotlib.org/>`_ (1.5.3 or higher, for plotting graphs
  and displaying images)
- `opencv <https://opencv.org/>`_ (3.0 or higher, to perform image acquisition
  and processing)
- `pyserial <https://pypi.org/project/pyserial/>`_ (To interface with serial
  sensors and actuators)
- `Tk <https://docs.python.org/3/library/tkinter.html>`_ (For the configuration
  interface of cameras)
- `scikit-image <https://scikit-image.org/>`_ (0.11 or higher)
- `Simple-ITK <https://simpleitk.org/>`_ (for faster image recording)
- `PyCUDA <https://documen.tician.de/pycuda/>`_ (for GPU accelerated features)

.. note::
  Knowing which modules are needed for a given setup is easy. Just write the
  script and start it, if a module is missing Crappy will simply tell you !

1. Check your Python version
----------------------------

Before installing Crappy, first check that you have **a compatible version of**
**Python** installed. You can get the current version of Python by running
:shell:`python --version` in a console. The version should then be displayed,
e.g. :shell:`Python 3.9.7`.

.. note::
  On Windows, Python is not natively installed and might not be present at
  all ! In this case, the given command will display an error message.

If the current version of Python is not compatible with Crappy (requires Python
>=3.6), or if Python is not installed, you will first need to **install a**
**compatible version of Python**. The precise installation steps for each OS
are beyond the scope of this documentation.

.. note::
  On Linux and MacOS, you will likely need to install the new version of Python
  alongside the original version. Never uninstall the original version, or your
  system will break !

2. Deploy a virtual environment (optional)
------------------------------------------

It is **recommended** to install Crappy in a `virtual environment
<https://docs.python.org/3/library/venv.html>`_, to avoid conflicts with other
Python packages installed at the user or system level. This step is however not
mandatory,and it is possible to install and run Crappy at the user level.

To create an virtual environment called `venv_crappy`, run the following
command at the location of your choice.

.. code-block:: shell-session

   python -m venv venv_crappy

This should create a new folder called `venv_crappy` at the location of your
console, containing an independent install of Python.

3. Install Crappy
-----------------

Once you have a compatible version of Python installed, and after optionally
setting up a virtual environment, you're **ready to install Crappy**. A single
line of code is necessary to install Crappy :

.. tabs::

   .. group-tab:: Without virtual environment

      .. code-block:: shell-session

         python -m pip install crappy

   .. group-tab:: In a virtual environment

      .. tabs::

         .. group-tab:: Linux & MacOS

            Assuming your console is at the location of the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy/bin/python -m pip install crappy

         .. group-tab:: Windows

            Assuming your console is at the location of the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy\Scripts\python.exe -m pip install crappy

Following th same pattern, you can also **install any additional module** that
you would need to use along with Crappy. For example :

.. code-block:: shell-session

   python -m pip install matplotlib

.. note::
  You can install at once most of the modules necessary for a specific use of
  Crappy by using the so-called extras. To do so, simply run :

  .. code-block:: shell-session

     python -m pip install crappy[<extra>]

  The available extras are ``SBC``, ``image``, ``hardware`` and ``main``. They
  contain respectively modules for interfacing with single board computers, for
  recording and displaying images and videos, for interfacing with hardware
  over serial or USB, and ``main`` contains the three most used modules in
  Crappy after the mandatory Numpy.

4. Check your install
---------------------

Once you have installed Crappy, you can **run a few checks** to make sure it
works fine on your system. First, try to simply import it :

.. tabs::

   .. group-tab:: Without virtual environment

      .. code-block:: shell-session

         python -c "import crappy;print(crappy.__version__)"

   .. group-tab:: In a virtual environment

      .. tabs::

         .. group-tab:: Linux & MacOS

            Assuming your console is at the location of the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy/bin/python -c "import crappy;print(crappy.__version__)"

         .. group-tab:: Windows

            Assuming your console is at the location of the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy\Scripts\python.exe -c "import crappy;print(crappy.__version__)"

This command should return without an error and print the installed version of
Crappy. If that is not the case, please refer to the :ref:`Troubleshooting`
page of the documentation.

If you can successfully import Crappy, you can then try to run a few examples
to confirm that Crappy operates as expected. The `examples folder
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples>`_ of
the GitHub repository contains a wide collection of readily-runnable examples.
To execute a test script called :file:`example.py`, run the following lines in
a console :

.. tabs::

   .. group-tab:: Without virtual environment

      .. code-block:: shell-session

         python example.py

   .. group-tab:: In a virtual environment

      .. tabs::

         .. group-tab:: Linux & MacOS

            Assuming your console is at the location of the virtual environment
            and that :file:`example.py` is at the same level as the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy/bin/python example.py

         .. group-tab:: Windows

            Assuming your console is at the location of the virtual environment
            and that :file:`example.py` is at the same level as the virtual
            environment :

            .. code-block:: shell-session

               venv_crappy\Scripts\python.exe example.py

If you're successful with all these steps, congratulations ! You just installed
Crappy on your machine ! We wish you success in your work.
