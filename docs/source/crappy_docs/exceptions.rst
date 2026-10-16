===================================
Exceptions & missing Python modules
===================================

.. autoexception:: crappy._exceptions.DefinitionError
.. autoexception:: crappy._exceptions.FailureTransportError
.. autoexception:: crappy._exceptions.GeneratorStop
.. autoexception:: crappy._exceptions.LinkDataError
.. autoexception:: crappy.tool.image_processing.LostPatchError
.. autoexception:: crappy.tool.image_processing.video_extenso.LostSpotError
.. autoexception:: crappy._exceptions.OptionalDependencyError
.. autoexception:: crappy._exceptions.ReaderStop
.. autoexception:: crappy._exceptions.SchedulerStop
.. autoexception:: crappy._exceptions.StartTimeout
.. autoexception:: crappy._exceptions.StopRequested
.. autoexception:: crappy._exceptions.T0NotSetError

.. autoclass:: crappy.OptionalModule
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
