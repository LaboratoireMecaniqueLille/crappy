# coding: utf-8

from typing import Dict, Type

from .meta_modifier import Modifier, MetaModifier

from .demux import Demux
from .differentiate import Diff
from .integrate import Integrate
from .mean import Mean
from .median import Median
from .moving_avg import Moving_avg
from .moving_med import Moving_med
from .offset import Offset
from .trig_on_change import Trig_on_change
from .trig_on_value import Trig_on_value

modifier_dict: Dict[str, Type[Modifier]] = MetaModifier.classes
