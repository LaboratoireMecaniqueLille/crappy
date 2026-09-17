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

Contributing drivers
--------------------

User contributions to ``crappy.collection`` are welcome. The aim is to build a
larger community-driven base of hardware drivers, including drivers written for
specialized devices or individual experimental setups.

Unlike contributions to the actively maintained parts of Crappy, collection
drivers are not expected to meet a particular code-quality standard. A driver
can therefore be shared with other users even when it is application-specific,
experimental, or not maintained over time.

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
   :members: name, kind, module, available, error_type, error_message
   :undoc-members:

.. autoclass:: crappy._collection.CollectionEntry
   :members: name, kind, module
   :undoc-members:

.. autoclass:: crappy._collection.CollectionUnavailableError
   :special-members: __init__
