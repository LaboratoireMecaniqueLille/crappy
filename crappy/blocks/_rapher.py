# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Grapher Grapher
# @{

## @file _grapher.py
# @brief The grapher plots data received from a block (via a Link).
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from _masterblock import MasterBlock
# Major library imports
from numpy import arange
from scipy.special import jn

# Enthought library imports
from enable.api import Window
from enable.example_support import DemoFrame, demo_main
from traits.api import HasTraits
from pyface.timer.api import Timer

# Chaco imports
from chaco.api import create_line_plot, OverlayPlotContainer
from chaco.tools.api import MoveTool, PanTool, ZoomTool

class Rapher(MasterBlock, ):
  """Plot the input data"""

  def __init__(self, *args, **kwargs):
    pass


