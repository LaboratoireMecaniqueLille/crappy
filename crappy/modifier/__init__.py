# coding: utf-8

from .modifier import Modifier, MetaModifier
from .apply_strain_img import Apply_strain_img
from .demux import Demux
from .differentiate import Diff
from .integrate import Integrate
from .mean import Mean
from .median import Median
from .moving_avg import Moving_avg
from .moving_med import Moving_med
from .trig_on_change import Trig_on_change
from .trig_on_value import Trig_on_value

modifier_list = MetaModifier.classes
