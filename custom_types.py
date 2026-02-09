from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from dynaplex.modelling import StateCategory, HorizonType


@dataclass(slots=True)
class NodeInfo:
    inventory_level: int  # positive = inventory, negative = backlog
    pipeline: List[int]


@dataclass(slots=True)
class SupplyChainState:
    node_infos: List[NodeInfo]
    remaining_time: int
    day: int
    category: StateCategory
    horizon_type: HorizonType
    num_actions: int


class PolicyType(Enum):
    BASE_STOCK = auto()
    FIXED_ORDER = auto()
    MIN_MAX = auto()
    FIXED_ORDER_QUANTITY = auto()

    
