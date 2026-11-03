===
API
===

This reference describes Crappy's public modules, classes, functions,
arguments, and return values. Each entry links to its source code when the
implementation provides useful additional detail.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Blocks and test control

      :doc:`crappy_docs/blocks` perform acquisition, commands, processing,
      display, and recording. :doc:`crappy_docs/aliases` includes functions
      for starting and stopping tests.

   .. grid-item-card:: Hardware objects

      :doc:`Camera <crappy_docs/cameras>`,
      :doc:`InOut <crappy_docs/inouts>`, and
      :doc:`Actuator <crappy_docs/actuators>` objects communicate with devices.

   .. grid-item-card:: Links and data transport

      :doc:`crappy_docs/links` carry labeled measurements and images between
      Blocks and define the connection rules.

   .. grid-item-card:: Modifiers

      :doc:`crappy_docs/modifiers` transform data as it passes through a Link.

   .. grid-item-card:: Tools and errors

      :doc:`crappy_docs/tools` provide reusable helpers.
      :doc:`crappy_docs/exceptions` documents errors raised by Crappy.

   .. grid-item-card:: Legacy and laboratory integrations

      :doc:`crappy_docs/collection` documents legacy drivers.
      :doc:`crappy_docs/lamcube` covers laboratory-specific integrations.

Complete API index
------------------

.. toctree::
   :maxdepth: 2

   crappy_docs/actuators.rst
   crappy_docs/collection.rst
   crappy_docs/blocks.rst
   crappy_docs/cameras.rst
   crappy_docs/modifiers.rst
   crappy_docs/inouts.rst
   crappy_docs/links.rst
   crappy_docs/tools.rst
   crappy_docs/exceptions.rst
   crappy_docs/aliases.rst
   crappy_docs/lamcube.rst
