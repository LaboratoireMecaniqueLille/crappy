# coding: utf-8

from .path import Path, ConditionType

paths_dict: dict[str, type[Path]] = Path.classes
