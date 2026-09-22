.. _concepts-image-pipelines:

===============
Image pipelines
===============

Crappy supports two architectures for acquiring and handling images. A
composable pipeline represents acquisition, processing, display, and recording
as separate VisionBlocks. The all-in-one architecture places those operations
behind one Camera Block.

.. important::

   VisionBlocks are recommended for new image pipelines. The all-in-one Camera
   Blocks remain supported and are not planned for deprecation.

Names used on this page
-----------------------

Several public objects contain “Camera” in their name but have different
roles:

- A :class:`crappy.camera.meta_camera.camera.Camera` is a hardware integration.
  It knows how to open a camera, expose its settings, acquire images, and close
  the device. It is called a **Camera object** on this page.
- A :class:`~crappy.blocks.vision.CameraSource` is a VisionBlock that owns one
  Camera object and publishes acquired images.
- A :class:`~crappy.blocks.vision.block.VisionBlock` is an independent image
  stage that can acquire, process, display, or record images.
- An :class:`~crappy.links.img_link.ImageLink` connects two VisionBlocks and
  exposes the newest image and matching metadata.
- A :class:`crappy.blocks.Camera` is the **all-in-one Camera Block**. It owns a
  Camera object and manages its image operations behind one Block in the
  script.

The two architectures
---------------------

.. graphviz:: ../diagrams/image_pipeline_architectures.dot
   :alt: A composable pipeline exposes CameraSource, processor, display, and recorder VisionBlocks connected by ImageLinks, while an all-in-one Camera Block contains acquisition and internal processing, display, and recording workers.
   :caption: Enclosures show ownership. In the composable pipeline, arrows labeled ImageLink connect public VisionBlocks. Dashed arrows labeled device API connect Camera objects to acquisition. The all-in-one Camera Block contains its complete supported fixed topology, with internal connections labeled shared latest frame.

The composable diagram shows the Camera object and acquisition operation inside
the CameraSource VisionBlock. CameraSource publishes the same acquired image to
a processing VisionBlock and an ImageRecorder. The processor publishes its
result to an ImageDisplayer. Each public stage appears in the script and can be
replaced, omitted, or connected to additional stages.

The enclosure on the right is the single all-in-one Camera Block visible in
the script. It contains the Camera object, acquisition operation, and internal
processing, display, and recording workers. Those workers are not independent
Blocks in the script's connection graph.

.. list-table:: Image architecture comparison
   :header-rows: 1
   :widths: 23 39 38

   * - Concern
     - VisionBlock pipeline
     - All-in-one Camera Block
   * - Script structure
     - Each image stage is an explicit Block connected by ImageLinks.
     - One Camera Block configures the available image operations.
   * - Acquisition
     - CameraSource owns and drives the Camera object.
     - The Camera Block owns and drives the Camera object.
   * - Processing
     - One or more processing VisionBlocks receive and publish images.
     - A Camera-specific internal worker performs the configured processing.
   * - Display and recording
     - ImageDisplayer and ImageRecorder are independent VisionBlocks.
     - Display and recording are managed internally by the Camera Block.
   * - Composition
     - Stages can be branched, reordered, reused, or run at different rates.
     - The public Block offers a compact interface to its predefined topology.
   * - Custom image stage
     - Subclass VisionBlock for a reusable stage in new pipelines.
     - Subclassing the internal CameraProcess family is supported for advanced
       all-in-one integrations.
   * - Recommended use
     - New image pipelines and workflows that benefit from explicit stages.
     - Existing scripts or workflows already matched by the fixed interface.

Building a VisionBlock pipeline
-------------------------------

A minimal composable pipeline uses a CameraSource and an ImageDisplayer:

.. code-block:: python

   source = crappy.blocks.vision.CameraSource("FakeCamera")
   display = crappy.blocks.vision.ImageDisplayer()
   crappy.img_link(source, display)

Adding recording does not require changing either existing Block. Create an
:class:`~crappy.blocks.vision.ImageRecorder` and connect a second ImageLink
from the source. Both consumers inspect the newest frame independently, so a
slower display does not by itself delay acquisition. The
:doc:`regular_links_and_image_links` page explains the resulting frame-skipping
behavior.

Each VisionBlock has a focused set of arguments and its own lifecycle. A
processing VisionBlock can publish a new image stream, send small numerical
results through regular Links, or do both. This makes explicit pipelines
suitable for custom processing and for workflows that branch into several
consumers.

Using an all-in-one Camera Block
--------------------------------

The all-in-one Camera Block remains useful when its available acquisition,
processing, display, and recording options already describe the intended
workflow. It exposes one Block in the script and manages the supporting image
workers internally.

Existing all-in-one scripts do not need to be rewritten merely because the
VisionBlock architecture is recommended for new work. Choose a migration only
when explicit stages solve a concrete need, such as adding an independent
consumer, reusing a custom processing stage, or rearranging the pipeline.

Both architectures use the same Camera objects for hardware integration and
can use a Camera configuration window. They differ in where image operations
are represented and managed, not in whether the underlying camera is
supported.

Public behavior and implementation details
------------------------------------------

The public distinction is between explicit VisionBlocks connected by
ImageLinks and one all-in-one Camera Block with a supported fixed interface.
The shared buffers, worker types, configuration channels, and synchronization
mechanisms are implementation details. Customization guides should use the
documented VisionBlock or CameraProcess hooks instead of accessing those
mechanisms directly.

API reference
-------------

- :class:`crappy.blocks.vision.CameraSource` acquires images in a composable
  pipeline.
- :class:`crappy.blocks.vision.block.VisionBlock` is the base for composable
  image stages.
- :class:`crappy.blocks.Camera` provides the all-in-one architecture.
- :class:`crappy.camera.meta_camera.camera.Camera` is the base for camera
  hardware integrations.
- :class:`crappy.blocks.camera_processes.CameraProcess` is the advanced base
  for all-in-one processing workers.
