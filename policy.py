from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from node import Node
from custom_types import State

@dataclass
class BasePolicy:
    """
    Abstract base class for inventory policies.
    Matches the DynaPlex policy interface.
    """
    node: Node

    def get_action(self, state: State) -> int:
        """Determines the action based on the current state."""
        raise NotImplementedError

    def set_parameters(self, **kwargs) -> None:
        """Updates policy hyperparameters."""
        raise NotImplementedError


@dataclass
class BaseStockPolicy(BasePolicy):
    """
    Base-stock policy with safety stock.
    Orders enough to reach (target + safety) inventory position.
    """
    target_inventory: int = 50
    safety_stock: int = 10
    price_per_unit: float = 20.0

    def set_parameters(
        self,
        target_inventory: int | None = None,
        safety_stock: int | None = None,
        price_per_unit: float | None = None,
    ) -> None:
        if target_inventory is not None:
            self.target_inventory = target_inventory
        if safety_stock is not None:
            self.safety_stock = safety_stock
        if price_per_unit is not None:
            self.price_per_unit = price_per_unit

    def get_action(self, state: State) -> int:
        # Accessing state.pipeline directly from the state object
        total_pending = sum(state.pipeline)
        
        # Calculate inventory position -> On-hand + On-order - Backlog
        inventory_position = state.inventory + total_pending - state.backorders
        base_stock_level = self.target_inventory + self.safety_stock
        
        # Order the difference, constrained by physical node capacity
        order_quantity = max(0, base_stock_level - inventory_position)
        return int(min(order_quantity, self.node.capacity - state.inventory))


@dataclass
class MinMaxPolicy(BasePolicy):
    """
    Min-max inventory policy (s, S policy).
    Orders up to max_inventory only when below min_inventory.
    """
    min_inventory: int = 20
    max_inventory: int = 80
    price_per_unit: float = 20.0

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_action(self, state: State) -> int:
        total_pending = sum(state.pipeline)
        inventory_position = state.inventory + total_pending

        if inventory_position < self.min_inventory:
            order_quantity = self.max_inventory - inventory_position
            return int(min(order_quantity, self.node.capacity - state.inventory))
        return 0


@dataclass
class FixedOrderPolicy(BasePolicy):
    """
    Fixed order quantity policy.
    Always orders a fixed amount if there is capacity.
    """
    order_quantity: int = 30
    price_per_unit: float = 20.0

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_action(self, state: State) -> int:
        total_pending = sum(state.pipeline)
        available_capacity = self.node.capacity - (state.inventory + total_pending)
        
        return int(min(self.order_quantity, max(0, available_capacity)))