.. _tutorial-custom-camera-configuration:

======================================
Customize Camera configuration windows
======================================

Add a **Confirm setup** checkbox that users must check before accepting the
camera configuration. The same custom
:class:`~crappy.blocks.vision.CameraSource` will work with either the Tkinter
or PyQt6 window.

Prerequisites
-------------

- Use Crappy 2.1.0 or later.
- Complete :doc:`image_pipeline` and be comfortable creating a Python subclass.
- Install Pillow and Matplotlib with the interpreter running the example:

  .. code-block:: shell-session

     python -m pip install Pillow matplotlib

- Use a graphical desktop. The default backend requires Tk support from the
  Python installation. For the PyQt6 backend, also install PyQt6:

  .. code-block:: shell-session

     python -m pip install PyQt6

The example uses :class:`~crappy.camera.FakeCamera`: no physical hardware or
output files are needed. It opens a configuration window, then displays images
for three seconds after you accept the setup. The configuration window has no
automatic timeout.

Create the custom configuration
-------------------------------

:download:`Download the complete script
</downloads/custom_objects/custom_camera_configuration.py>`, or save this
code as ``custom_camera_configuration.py``:

.. literalinclude:: /downloads/custom_objects/custom_camera_configuration.py
   :language: python
   :start-after: # [custom-camera-configuration-start]
   :end-before: # [custom-camera-configuration-end]

Run and accept the configuration
--------------------------------

From the directory containing the script, run:

.. code-block:: shell-session

   python custom_camera_configuration.py

In the configuration window:

1. Check **Confirm setup**.
2. Click **Apply Settings**.
3. Close the window.

Closing before applying the checkbox leaves the window open with an
explanation. You can instead enable **Auto apply** before checking it.

The terminal reports ``Camera setup confirmed``. The image window then shows
:class:`~crappy.camera.FakeCamera`'s moving grayscale pattern. A
``Stop criterion reached`` warning indicates the planned end of the experiment.

To try the same confirmation with PyQt6, change ``config_backend='tkinter'`` to
``config_backend='pyqt'`` in the ``ConfirmedCameraSource`` call in ``main()``,
then run the script again with the same command.

Adapt the rule to your experiment
---------------------------------

To add your own setting or acceptance condition, override these
:class:`~crappy.tool.camera_config.base.camera_config.CameraConfig` methods in
``ConfirmedConfig``:

- In :meth:`_create_local_settings() <crappy.tool.camera_config.base.\
  camera_config.CameraConfig._create_local_settings>`, create the additional
  setting and return it alongside the parent's settings. These settings belong
  to the window, so you do not need to modify the
  :class:`~crappy.camera.meta_camera.camera.Camera` driver.
- In :meth:`_validate_close() <crappy.tool.camera_config.base.camera_config.\
  CameraConfig._validate_close>`, return a message to keep the window open, or
  :obj:`None` to accept the setup. Check the parent's result first so existing
  selection requirements still apply.
- In :meth:`_on_valid_close() <crappy.tool.camera_config.base.camera_config.\
  CameraConfig._on_valid_close>`, add any action to perform after acceptance.
  The example logs ``Camera setup confirmed``.

Keep the :obj:`super() <super>` calls shown in the example. Combine the shared
class with each backend as shown by ``ConfirmedTkConfig`` and
``ConfirmedQtConfig``, putting the shared class first.

Use ``ConfirmedCameraSource`` in place of
:class:`~crappy.blocks.vision.CameraSource` in your pipeline, supplying your
usual camera arguments. Its ``configurator`` mapping selects the custom
window for the source's ``config_backend``. For an all-in-one Camera Block,
inherit :class:`crappy.blocks.Camera` instead and keep the same mapping.

Keep custom classes at module scope, and create
:class:`Blocks <crappy.blocks.meta_block.block.Block>`, connect
:class:`Links <crappy.links.link.Link>`, and call ``crappy.start()`` inside the
main guard, as in the complete script.

Change only one backend
-----------------------

To give only the Qt window a different title, add these classes above
``main()`` in the example:

.. code-block:: python

   class LaboratoryQtConfig(ConfirmedQtConfig):
     def _set_layout(self):
       super()._set_layout()
       self.setWindowTitle('Laboratory Camera setup')

   class LaboratoryCameraSource(ConfirmedCameraSource):
     configurator = {'tkinter': ConfirmedTkConfig,
                     'pyqt': LaboratoryQtConfig}

Replace ``ConfirmedCameraSource`` with ``LaboratoryCameraSource`` in
``main()`` and set ``config_backend='pyqt'``, then run the script again. The
window title becomes **Laboratory Camera setup**, and the confirmation checkbox
still works. Set ``config_backend='tkinter'`` to keep the original Tkinter
window.

For a Tkinter-only title change, subclass ``ConfirmedTkConfig`` instead and
call ``self.title('Laboratory Camera setup')`` after the parent's
``_set_layout()``. Use the corresponding backend subclass for other widget or
input-binding changes. If overriding ``__init__()``, forward its arguments to
``super().__init__()`` and create widgets only after that call returns.

Extend a specialized selection
------------------------------

To customize a processor's selection window, set ``configurator`` on a subclass
of that processor rather than on the image source. For example, add a log
message when users accept their :class:`~crappy.blocks.vision.DICVEProcessor`
patches:

.. code-block:: python

   import logging

   import crappy
   from crappy.tool.camera_config.base import DICVEConfig
   from crappy.tool.camera_config.tkinter import TkinterDICVEConfig
   from crappy.tool.camera_config.pyqt import PyQtDICVEConfig

   class LoggedPatches(DICVEConfig):
     def _on_valid_close(self):
       super()._on_valid_close()
       self.log(logging.INFO, 'Tracking patches confirmed')

   class LoggedTkPatches(LoggedPatches, TkinterDICVEConfig):
     pass

   class LoggedQtPatches(LoggedPatches, PyQtDICVEConfig):
     pass

   class LoggedDICVE(crappy.blocks.vision.DICVEProcessor):
     configurator = {'tkinter': LoggedTkPatches, 'pyqt': LoggedQtPatches}

Use ``LoggedDICVE`` instead of :class:`~crappy.blocks.vision.DICVEProcessor`
when creating your processor. Select the GUI backend on
:class:`~crappy.blocks.vision.CameraSource`, select patches in the window, and
close it. The terminal reports ``Tracking patches confirmed`` before processing
starts. The :obj:`super() <super>` call preserves the usual patch finalization.
For an all-in-one pipeline, inherit :class:`crappy.blocks.DICVE` instead.

For another algorithm, use its matching shared and backend classes, such as
:class:`~crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig`
or :class:`~crappy.tool.camera_config.base.video_extenso_config.\
VideoExtensoConfig`. Leave
:meth:`get_config() <crappy.tool.camera_config.base.camera_config.\
CameraConfig.get_config>` unchanged unless you also adapt the processor that
receives its result.

See :doc:`../crappy_docs/tools` for the available configuration classes and
hooks. For the design behind these extension points, see
:ref:`architecture-camera-configuration`.
