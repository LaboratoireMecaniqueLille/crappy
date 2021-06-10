=============
Documentation
=============

The latest version for the branch master can be accessed at: https://laboratoiremecaniquelille.github.io/crappy/

To build it yourself, first install ``sphinx``. The steps for the most common OS are detailed here: https://www.sphinx-doc.org/en/master/usage/installation.html

Then open a terminal in the ``/docs/`` directory of crappy and run: ::

	make html

Information about the documentation building is being displayed, including any warnings or errors.
The documentation files, in `.html` format, can then be found in ``/docs/build/html/``.

Finally simply double-click on any `.html` file, it should open in a web browser ! 
From there you'll have access to all of the documentation, regardless of the file you chose to open.

.. Note::	The documentation is computed partly from the `.rst` files in ``/docs/``, and partly from the docstrings of the `.py` files. The docstrings in these files should be in Google style.
