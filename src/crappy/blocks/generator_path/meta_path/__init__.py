# coding: utf-8

from .path import Path, ConditionType
from .meta_path import MetaPath

paths_dict: dict[str, type[Path]] = MetaPath.classes
