from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from dynaplex.modelling import StateCategory


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
    current_node_index: int

    # For multi-node I found no other way of tracking pending orders than to add it to the state

    # Is it neccesary to have pending orders in the state in this way? This list can grow large with a high number of nodes I think.
    # The inventory position is calculated as on-hand inventory + pending orders - backlog

    pending_orders: List[int] 


class PolicyType(Enum):
    BASE_STOCK = auto()
    FIXED_ORDER = auto()
    MIN_MAX = auto()
    FIXED_ORDER_QUANTITY = auto()

    
