=============
Core concepts
=============

These pages explain how the main parts of Crappy fit together. Tutorials show
how to complete a task, while this section explains the behavior that is
shared by different tasks. The current implementation details for contributors
are documented in the :doc:`architecture` guide.

Start with :doc:`concepts/blocks_links_labels` if you are new to Crappy. Read
:doc:`concepts/lifecycle_shutdown` when writing a custom Block or investigating
how a test starts and stops. The remaining pages explain the two data transport
types, compare the image-pipeline architectures, and help you choose a custom
object type.

.. toctree::
   :maxdepth: 1

   Blocks, Links, and labels <concepts/blocks_links_labels>
   Lifecycle and shutdown <concepts/lifecycle_shutdown>
   Regular Links and ImageLinks <concepts/regular_links_and_image_links>
   Image pipeline architectures <concepts/image_pipelines>
   Choose a custom object type <concepts/choosing_custom_object_type>
