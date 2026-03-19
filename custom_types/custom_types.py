from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from dynaplex.modelling import StateCategory
from dynaplex_playgroud.data.action_base import Action
import torch


@dataclass(slots=True)
class NodeInfo:
    inventory_level: int  # positive = inventory, negative = backlog
    pipeline: List[int] # Units that have already been shipped and are physically traveling toward the node


@dataclass(slots=True)
class SupplyChainState:
    
    node_infos: List[NodeInfo]
    day: int
    category: StateCategory
    current_node_index: int
    remaining_time: int

    # For multi-node I found no other way of tracking pending orders than to add it to the state

    # Is it neccesary to have pending orders in the state in this way? This list can grow large with a high number of nodes I think.
    # The inventory position is calculated as on-hand inventory + pending orders - backlog

    pending_orders: List[int] # How many units node i has ordered from its upstream supplier but that upstream has not yet shipped


@dataclass(slots=True)
class ReorderAction:
    order_quantity: int  


@dataclass
class GraphAction(Action):
    order_quantity: int


@dataclass(slots=True)
class SCGlobalState:
    node_features: torch.Tensor  # [node_features, current_node_index, category] becasue The dynaplex sequence builder needs fixed-size fields it can inspect statically
    current_node_index: float
    remaining_time: float
    category: StateCategory


class PolicyType(Enum):
    BASE_STOCK = auto()
    FIXED_ORDER = auto()
    MIN_MAX = auto()
    FIXED_ORDER_QUANTITY = auto()