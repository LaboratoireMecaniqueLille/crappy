# coding: utf-8
##  @addtogroup links
# @{

##  @defgroup MetaCondition MetaCondition
# @{

## @file _metaCondition.py
# @brief Metaclass for all Links conditions. Must implement the evaluate method.
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

import abc


class MetaCondition:
  """
  Metaclass for all Links conditions. Must implement the evaluate method.
  """

  def __init__(self):
    pass

  __masterblockclass__ = abc.ABCMeta

  @abc.abstractmethod
  def evaluate(self):
    """
    This method is called by the Links and must always be implemented.
    """
    pass
