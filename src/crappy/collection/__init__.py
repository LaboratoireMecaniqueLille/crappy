# coding: utf-8

from .manifest import DRIVERS
from .api import check, check_all, drivers
from .._collection import collection_registry

# Register all the drivers discovered in the local manifests
collection_registry.register(*DRIVERS)
