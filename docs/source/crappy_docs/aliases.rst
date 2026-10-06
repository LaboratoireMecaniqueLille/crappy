=======
Aliases
=======

Link Methods
------------
.. autofunction:: crappy.link

.. autofunction:: crappy.img_link

.. autofunction:: crappy.display_graph

Open Online Documentation
-------------------------
.. autofunction:: crappy.docs

Packaged example resources
--------------------------

``crappy.resources`` provides the example images distributed with Crappy.
These public helpers are intended for examples, demonstrations, and custom
scripts that need the same reference images.

.. autoclass:: crappy.resources

.. py:attribute:: crappy.resources.speckle

   Greyscale speckle image as a NumPy array. Loading the array requires
   OpenCV.

.. py:attribute:: crappy.resources.ve_markers

   Greyscale image containing video-extensometry markers as a NumPy array.
   Loading the array requires OpenCV.

.. py:attribute:: crappy.resources.pad

   Canvas demonstration image as a NumPy array. Loading the array requires
   OpenCV.

.. py:attribute:: crappy.resources.paths

   Mapping from ``'speckle'``, ``'ve_markers'``, and ``'pad'`` to their
   packaged image paths. Use these paths when another library should load the
   images.

Class aliases
-------------
.. autoclass:: crappy.Actuator
   :noindex:
.. autoclass:: crappy.Block
   :noindex:
.. autoclass:: crappy.VisionBlock
   :noindex:
.. autoclass:: crappy.Camera
   :noindex:
.. autoclass:: crappy.InOut
   :noindex:
.. autoclass:: crappy.Modifier
   :noindex:
.. autoclass:: crappy.Path
   :noindex:

Method aliases
--------------

crappy.prepare()
++++++++++++++++
.. automethod:: crappy.Block.prepare_all
   :noindex:

crappy.renice()
+++++++++++++++
.. automethod:: crappy.Block.renice_all
   :noindex:

crappy.launch()
+++++++++++++++
.. automethod:: crappy.Block.launch_all
   :noindex:

crappy.start()
++++++++++++++
.. automethod:: crappy.Block.start_all
   :noindex:

crappy.stop()
+++++++++++++
.. automethod:: crappy.Block.stop_all
   :noindex:

crappy.reset()
++++++++++++++
.. automethod:: crappy.Block.reset
   :noindex:
