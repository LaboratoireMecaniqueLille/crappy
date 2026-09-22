======================
Developers information
======================

.. role:: py(code)
  :language: python
  :class: highlight

Contributing to Crappy
----------------------

This page explains how to contribute to Crappy, build its documentation, and
understand its runtime architecture.

Submit improvements through a `GitHub pull request
<https://github.com/LaboratoireMecaniqueLille/crappy/compare>`_. Follow these
project conventions:

- Follow `PEP8 <https://peps.python.org/pep-0008/>`_ as much as possible,
  except for the indents that we chose to lower from 4 to 2 spaces for
  compactness.

- Use the `Google style <https://google.github.io/styleguide/pyguide.html>`_
  for docstrings. Please comment and document your code extensively, and
  update the documentation when behavior changes.

- Use relevant and meaningful titles and descriptions for your commits.
  Starting from v2.0.0, the rules `described here
  <https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/>`_
  should be used for commit messages.

The development branch of Crappy is called `develop
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/develop>`_, and is
the one on which you should commit. Starting from v2.0.0, the `master branch
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master>`_ is never
directly committed to.

Documentation style
-------------------

Write documentation in American English with a calm, technical, and direct
voice. Lead with what an object or procedure does, when to use it, and any
constraints that affect the reader.

Apply these rules to all documentation changes:

- Use second person for instructions and neutral language for behavior and
  architecture.
- Prefer observable behavior over claims such as “easy”, “powerful”, or
  “optimal”. Qualify performance and timing statements with a benchmark or an
  implementation guarantee.
- Do not use semicolons, rhetorical questions, congratulatory endings, or
  exclamation marks in prose.
- Use bold text for warnings and defined terms, not for ordinary emphasis.
- Expand an abbreviation on its first use on each page.
- Keep sentences concise and put lifecycle, timing, multiprocessing, safety,
  and cleanup constraints next to the first relevant step.
- End tutorials with links to specific next tasks or API entries instead of a
  generic summary.
- Do not add ``sectionauthor`` directives. Git history records authorship.
- Add technical names that a spell checker does not recognize to
  ``docs/spelling_wordlist.txt``. Do not add ordinary misspellings.

Use qualified Camera terminology when a name could be ambiguous:

- :class:`crappy.camera.meta_camera.camera.Camera` is the base class for camera
  hardware integrations. After the first mention, call it a “Camera object”.
- :class:`crappy.blocks.Camera` is the supported all-in-one Camera Block.
- :class:`crappy.blocks.vision.CameraSource` acquires images in a composable
  image pipeline.
- :class:`crappy.blocks.vision.block.VisionBlock` is the base class for
  composable image stages.
- :class:`crappy.links.link.Link` carries dictionaries between Blocks.
- :class:`crappy.links.img_link.ImageLink` shares the newest coherent image
  between VisionBlocks.

VisionBlocks are recommended for new image pipelines. The all-in-one Camera
Blocks remain supported and are not planned for deprecation. Do not describe
the all-in-one architecture as obsolete, superseded, or deprecated.

A tutorial or how-to guide should begin with its outcome, prerequisites,
required hardware, and runtime side effects such as opening a graphical user
interface (GUI) or writing files. State version requirements when behavior is
version-sensitive. Provide complete examples as plain text, even when the same
content is also available as a download.

After installing ``codespell``, run the editorial spelling check from the
repository root:

.. code-block:: console

   $ codespell docs/source docs/README.md --skip='*.py,*.cu' \
       --ignore-words=docs/spelling_wordlist.txt

Building the documentation
--------------------------

Build the documentation in a clean virtual environment with a supported
Python version. From the repository root, run:

.. code-block:: console

   $ python -m venv .venv-docs
   $ source .venv-docs/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install -r docs/source/requirements.txt
   $ python -m pip install .
   $ make -C docs clean html

The final command performs a fresh, nitpicky Sphinx build and treats warnings
as errors. It also regenerates the architecture diagrams from the versioned
Graphviz ``.dot`` sources. The generated site is available in
``docs/build/html``.

The documentation requirements constrain Sphinx to one feature release line
shared by all supported Python versions and pin each direct third-party
extension. Crappy's runtime dependencies, including NumPy, are resolved from
``pyproject.toml``. Review the Sphinx constraint and extension pins together
when updating the documentation toolchain.

Runtime concepts and architecture
---------------------------------

The user-facing runtime contracts now live in the :doc:`concepts` section.
Use these pages when a tutorial or API entry needs to explain shared behavior:

- :doc:`concepts/blocks_links_labels`
- :doc:`concepts/lifecycle_shutdown`
- :doc:`concepts/regular_links_and_image_links`
- :doc:`concepts/image_pipelines`
- :doc:`concepts/choosing_custom_object_type`

The :doc:`architecture` guide documents process ownership, startup
coordination, transport internals, error propagation, cleanup, and the
all-in-one Camera internals for contributors. Keep details that may change
without affecting application code in that guide rather than in tutorials.
