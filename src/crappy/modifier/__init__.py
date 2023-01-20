# coding: utf-8

from typing import Dict, Type

from .meta_modifier import Modifier, MetaModifier

from .demux import Demux
from .differentiate import Diff
from .integrate import Integrate
from .mean import Mean
from .median import Median
from .moving_avg import MovingAvg
from .moving_med import MovingMed
from .offset import Offset
from .trig_on_change import TrigOnChange
from .trig_on_value import TrigOnValue

modifier_dict: Dict[str, Type[Modifier]] = MetaModifier.classes
