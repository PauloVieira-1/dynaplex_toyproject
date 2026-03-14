from __future__ import annotations
from dataclasses import dataclass
from custom_types import SupplyChainState, Node
import random as rd


@dataclass
class BasePolicy:
    node: Node

    def get_action(self, state: SupplyChainState) -> int:
        raise NotImplementedError

    def set_parameters(self, **kwargs) -> None:
        raise NotImplementedError


# Base Stock Policy 
# -------------------

@dataclass
class BaseStockPolicy(BasePolicy):

    # Default parameters and should be adjusted later
    target_inventory: int = 50
    safety_stock: int = 10
    price_per_unit: float = 20.0

    def get_action(self, node_info) -> int:

        total_pending = sum(node_info.pipeline)

        # Inventory position = on-hand - backlog + pending (see SupplyChainState)
        inventory_position = node_info.inventory_level + total_pending
        base_stock_level = self.target_inventory + self.safety_stock

        order_quantity = max(0, base_stock_level - inventory_position)

        on_hand = max(0, node_info.inventory_level)
        max_feasible = self.node.capacity - on_hand

        return int(min(order_quantity, max(0, max_feasible)))


# Min-Max Policy (s, S)
# -------------------

@dataclass
class MinMaxPolicy(BasePolicy):

    min_inventory: int = 20
    max_inventory: int = 80
    price_per_unit: float = 20.0

    def get_action(self, node_info) -> int:

        total_pending = sum(node_info.pipeline)
        inventory_position = node_info.inventory_level + total_pending

        if inventory_position < self.min_inventory:
            order_quantity = self.max_inventory - inventory_position

            on_hand = max(0, node_info.inventory_level)
            max_feasible = self.node.capacity - on_hand

            return int(min(order_quantity, max(0, max_feasible)))

        return 0


# Fixed Order Policy
# -------------------

@dataclass
class FixedOrderPolicy(BasePolicy):

    order_quantity: int = 30
    price_per_unit: float = 20.0 # Should be sey by noode (change later)

    def get_action(self, node_info) -> int:
        available_capacity = self.node.capacity - max(0, node_info.inventory_level)
        return int(min(self.order_quantity, max(0, available_capacity)))


# Random Choice Policy
# -------------------

class RandomChoice(BasePolicy):

    price_per_unit: float = 20.0

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_action(self, node_info) -> int:

        available_capacity = self.node.capacity - max(0, node_info.inventory_level)
        random_quantity = rd.randrange(0, available_capacity + 1 )

        return random_quantity