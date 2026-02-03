from enum import Enum, auto
from dataclasses import dataclass


class StateCategory(Enum):
    AWAIT_EVENT = auto()
    AWAIT_ACTION = auto()
    FINAL = auto()

@dataclass(slots=True)
class State:
    """
    Dynamic state for a single supply-chain node.
    """
    inventory: int
    backorders: int
    remaining_time: int
    day: int
    pipeline: list[int]
    category: StateCategory = StateCategory.AWAIT_EVENT


class PolicyType(Enum):
    BASE_STOCK = auto()
    FIXED_ORDER = auto()
    MIN_MAX = auto()
    FIXED_ORDER_QUANTITY = auto()

    
