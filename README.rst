=======================
Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=======================

This package aims to provide easy-to-use tools for command and acquisition on 
complex experimental setups.

.. contents::

Description
-----------

See ``DESCRIPTION.rst`` for a more complete description.


Installation
------------

Only tested on Ubuntu 14.04 / 15.1 :

       git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
       
       cd crappy

       sudo python setup install


Documentation
-------------

The documentation can be compiled by simply running in the doc folder :

       make html

Note that you will need to install sphinx and numpy-doc:
       
       pip install -U Sphinx
       
       apt-get install python-numpydoc
        
We used the scipy-style for Sphinx, you can find the original on SciPy's Github:

https://github.com/scipy/scipy


Bug reports
-----------

Please reports bugs at:

https://github.com/LaboratoireMecaniqueLille/crappy/issues


License information
-------------------

See the file ``LICENSE.txt`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
