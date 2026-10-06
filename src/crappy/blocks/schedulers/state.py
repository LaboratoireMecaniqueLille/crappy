# coding: utf-8

from collections.abc import Sequence, Callable
from typing import Any

SchedCondType = tuple[Callable[[float,  dict[str, list[Any]]], bool], str]
SchedOutType = Callable[[float,  dict[str, list[Any]]], dict[str, Any] | None]


class State:
  """Describes one state in a :class:`~crappy.blocks.Scheduler` workflow.

  A State associates output functions with ordered stop conditions. While the
  State is active, its stop conditions determine when the Scheduler switches
  to another State. If no stop condition is met, its output functions generate
  the values to send to downstream Blocks.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               id_: str,
               outputs: Sequence[SchedOutType],
               stop_conditions: Sequence[SchedCondType]) -> None:
    """Sets the State identifier, outputs and stop conditions.

    Args:
      id_: The identifier of the State, as a non-empty :obj:`str`. It must be
        unique among the States given to a Scheduler. The identifiers
        ``'Start'`` and ``'End'`` are reserved by the Scheduler.
      outputs: A sequence of callables that generate this State's output
        values. Each callable receives the time elapsed since the State was
        entered and the data received from upstream Blocks. It should return a
        :obj:`dict` mapping output labels to values, or :obj:`None` when it has
        nothing to output. All the callables are evaluated in sequence, and a
        value returned by a later callable overwrites an earlier value for the
        same label.
      stop_conditions: A sequence of pairs containing a callable and a target
        State identifier. Each callable receives the time elapsed since the
        State was entered and the data received from upstream Blocks, and
        should return a :obj:`bool`. Conditions are evaluated in sequence, and
        the first true condition causes a transition to its target State. A
        target of ``'End'`` terminates the workflow.
    """

    match id_:
      case str() if id_.strip():
        self.id: str = id_
      case str():
        raise ValueError("The State index must be provided as a non-empty "
                         "string")
      case _:
        raise TypeError("The State index must be provided as a non-empty "
                        "string")
    match outputs:
      case (*outputs,) if all(callable(out) for out in outputs):
        self.outputs: tuple[SchedOutType, ...] = tuple(outputs)
      case _:
        raise TypeError("The State outputs must be provided as a sequence of "
                        "callables")
    match stop_conditions:
      case (*stop,) if (all(isinstance(stop_cond, Sequence) and
                            not isinstance(stop_cond, (str, bytes)) and
                            len(stop_cond) == 2
                            for stop_cond in stop) and
                        all(callable(cond) and isinstance(label, str)
                            and label.strip() for cond, label in stop)):
        self.stop_conditions: tuple[SchedCondType,
                                    ...] = tuple(stop_conditions)
      case _:
        raise TypeError("The State stop conditions must be provided as a "
                        "sequence of tuples containing each a callable and a"
                        "non-empty label as a string")
