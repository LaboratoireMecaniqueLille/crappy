===================================
Exceptions & missing Python modules
===================================

.. automodule:: crappy._exceptions

See :ref:`concepts-lifecycle-shutdown` for handling failures from a Crappy run.
The classes below describe individual failures and their original details.

.. autoexception:: crappy._exceptions.DefinitionError
   :special-members: __init__

.. autoexception:: crappy._exceptions.FailureTransportError
   :special-members: __init__

.. autoexception:: crappy._exceptions.GeneratorStop
.. autoexception:: crappy._exceptions.LinkDataError
.. autoexception:: crappy.tool.image_processing.LostPatchError
.. autoexception:: crappy.tool.image_processing.video_extenso.LostSpotError
.. autoexception:: crappy._exceptions.OptionalDependencyError
   :special-members: __init__

.. autoexception:: crappy._exceptions.ProcessExitError
   :members:
   :special-members: __init__

.. autoexception:: crappy._exceptions.ShutdownTimeoutError
   :members:
   :special-members: __init__

.. autoclass:: crappy._exceptions.RemoteFailureMetadata
   :members: from_report
   :special-members: __init__

The remote replacement exceptions below use the constructor arguments described
by :class:`~crappy._exceptions.RemoteFailureMetadata`. They preserve the
available failure details when the original exception type cannot be restored.

.. autoexception:: crappy._exceptions.RemoteFailure
.. autoexception:: crappy._exceptions.RemoteKeyboardInterrupt
.. autoexception:: crappy._exceptions.RemoteSystemExit
.. autoexception:: crappy._exceptions.RemoteBaseFailure
.. autoexception:: crappy._exceptions.ReaderStop
.. autoexception:: crappy._exceptions.SchedulerStop
.. autoexception:: crappy._exceptions.StartTimeout
.. autoexception:: crappy._exceptions.StopRequested
.. autoexception:: crappy._exceptions.T0NotSetError

.. autoclass:: crappy.OptionalModule
   :members: resolve, is_available
   :special-members: __init__, __getattr__, __call__

.. autoclass:: crappy._failure.ExceptionNode

.. autoclass:: crappy._failure.FailureReport
   :members: to_wire, from_wire
   :special-members: __str__

.. autoclass:: crappy._failure.FailureManager
   :members: set_queue, receive, collect, step, capture_exception, forward,
             snapshot, raise_if_failed, record_context_exception,
             iter_exception_members
   :special-members: __init__
