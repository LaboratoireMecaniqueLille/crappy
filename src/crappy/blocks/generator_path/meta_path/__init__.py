# coding: utf-8

from typing import Dict, Type

from .path import Path, ConditionType
from .meta_path import MetaPath

paths_dict: Dict[str, Type[Path]] = MetaPath.classes
