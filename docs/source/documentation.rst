=============
Documentation
=============

Access online documentation
---------------------------

The latest version of Crappy's code can be found on `GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy>`_, and the latest version
of Crappy's documentation can be found on `ReadTheDocs
<https://crappy.readthedocs.io/en/latest/>`_.

An easy way to access the online documentation is to run in a Python console :

  >>> import crappy
  >>> crappy.doc()

This will open the documentation in a new tab or window of your default browser,
hard to be quicker than that ! Of course you still need an internet connection.

Build local documentation
-------------------------

Required packages
+++++++++++++++++

All the documentation is actually contained in Crappy's ``.py`` files in the
form of docstrings and in the ``.rst`` files. It is thus possible to build the
documentation locally as soon as Crappy is installed.
To build it yourself, first install ``sphinx``. This can be done easily using
``pip``, more detail `here
<https://www.sphinx-doc.org/en/master/usage/installation.html>`_. You will also
need ``sphinx_rtd_theme``, which is distributed on `PyPI
<https://pypi.org/project/sphinx-rtd-theme/>`_ and should also be installed via
``pip``.

Getting the doc files
+++++++++++++++++++++

Then, you need to get the ``.rst`` files and the Makefile. There are several
options :

- Regardless of how Crappy was installed, it is possible to download the
  Makefile and ``.rst`` files from the `Github repository
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/docs>`_.

- If you installed Crappy using ``git`` and ``setup``, then the Makefile and
  ``.rst`` files are located in the ``/<install_path>/crappy/docs/`` folder
  (or ``C:\<install_path>\crappy\docs\`` on Windows).

- If you installed Crappy in a virtualenv, the Makefile and ``.rst`` files
  are located in your virtualenv folder in ``/crappy/docs/``.

- If you installed Crappy via ``pip`` on your system, there's no guarantee
  on where the Makefile and ``.rst`` files have been copied by ``pip``.
  Downloading them is probably the easiest way to go. It is nevertheless for
  sure that they're somewhere in a ``/crappy/docs/`` folder
  (``\crappy\docs\`` on Windows). From our own experience, this folder may
  be located :

    - In ``/home/<username>/.local/`` in Linux (tested on Ubuntu 20.04).

    - In ``C:\Users\<username>\AppData\Local\Programs\Python\Pythonxy\`` in
      Windows, x.y being the installed Python version (tested on Windows
      10).

    - In ``/Library/Frameworks/Python.framework/Versions/Current/`` in macOS
      (tested on mac Sierra).

Building the documentation
++++++++++++++++++++++++++

Once you located or downloaded the Makefile and ``.rst`` files, simply open a
terminal in the directory where Makefile is. Assuming that the folder containing
the ``.rst`` files is located in the same directory and called ``source``,
simply run :

.. code-block:: shell-session

  sphinx-build -b html source build

This will create a new directory ``build``, containing the ``.html``
documentation files. These files can be opened using your regular web browser.

Note that the building process can be tuned in several ways (format, folders
path and names, etc.), see `Sphinx documentation
<https://www.sphinx-doc.org/en/master/usage/quickstart.html#running-the-build>`_
for details.

**For Linux users**, running :

.. code-block:: shell-session

  make html

Is a simpler way of calling the building command outputting ``.html`` files with
the default arguments.
