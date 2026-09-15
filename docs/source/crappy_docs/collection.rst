=================
Driver collection
=================

``crappy.collection`` contains hardware drivers that have been kept for
backward compatibility, or just shared by the community, but are not actively
maintained. They are by default not imported with Crappy.

Before using one of these drivers in a Block, explicitly import the collection
package once near the beginning of the script:

.. code-block:: python

   import crappy
   import crappy.collection

The driver can then be selected by its usual class name, for example:

.. code-block:: python

   sensor = crappy.blocks.IOBlock("Comedi", channels=[0])

The Actuator, Camera, and InOut API pages still contain the class-level
documentation for collection drivers. Their import paths start with
``crappy.collection`` to distinguish them from actively maintained drivers.

Discovering and checking drivers
--------------------------------

Listing the collection classes does not import the concrete driver modules or
their optional dependencies:

.. autofunction:: crappy.collection.drivers

The following helpers attempt to import one or all the drivers declared in
collection, and report either successful loading or import failures:

.. autofunction:: crappy.collection.check

.. autofunction:: crappy.collection.check_all

Below are the classes on which the collection import machinery is relying:

.. autoclass:: crappy.collection.api.CheckResult
   :members:

.. autoclass:: crappy._collection.CollectionEntry
   :members:

.. autoclass:: crappy._collection.CollectionUnavailableError
   :members:
