import sys
from dataclasses import dataclass

from prorl.common.enum_utils import ExtendedEnum

if sys.version_info >= (3, 8):
    from typing import Dict, Optional, TypedDict
else:
    from typing import Dict, Optional
    from typing_extensions import TypedDict


class ResourceClass(TypedDict):
    cost: float
    capacity: float
    allocated: int


class ResourceDistribution(str, ExtendedEnum):
    Equally = 'equally'
    AllAndNothing = 'all-and-nothing'
    Pool = 'pool'


@dataclass
class EnvResource:
    name: str
    bucket_size: int
    total_available: int
    allocated: int
    units_allocated: int = 0
    total_units: int = 0
    min_buckets: int = 0
    classes: Optional[Dict[str, ResourceClass]] = None


@dataclass
class NodeResourceValue:
    name: str
    allocated: int
    classes: Optional[Dict[str, ResourceClass]] = None


@dataclass
class NodeResource:
    name: str
    bucket_size: int
    allocated: int
    min_buckets: int = 0
    classes: Optional[Dict[str, ResourceClass]] = None


class NodeTypeDistribution(str, ExtendedEnum):
    Random = 'random'
    Equally = 'equally'
