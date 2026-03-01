from __future__ import annotations
from dataclasses import dataclass
from custom_types import SupplyChainState, Node


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
    """
    Base-stock policy with safety stock.
    Orders up to (target + safety) inventory position.
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

    def get_action(self, node_info) -> int:

        total_pending = sum(node_info.pipeline)

        # Inventory position = on-hand - backlog + on-order
        inventory_position = node_info.inventory_level + total_pending - max(0, node_info.inventory_level)  # Backlog is negative inventory, so subtract it out
        base_stock_level = self.target_inventory + self.safety_stock

        order_quantity = max(0, base_stock_level - inventory_position)

        # Respect capacity constraint
        on_hand = max(0, node_info.inventory_level)
        max_feasible = self.node.capacity - on_hand

        return int(min(order_quantity, max_feasible))


# Min-Max Policy (s, S)
# -------------------

@dataclass
class MinMaxPolicy(BasePolicy):
    """
    Min-max (s, S) policy.
    Orders up to S when inventory position falls below s.
    """
    min_inventory: int = 20
    max_inventory: int = 80
    price_per_unit: float = 20.0

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_action(self, node_info) -> int:

        total_pending = sum(node_info.pipeline)
        inventory_position = node_info.inventory_level + total_pending

        if inventory_position < self.min_inventory:
            order_quantity = self.max_inventory - inventory_position

            on_hand = max(0, node_info.inventory_level)
            max_feasible = self.node.capacity - on_hand

            return int(min(order_quantity, max_feasible))

        return 0


# Fixed Order Policy
# -------------------

@dataclass
class FixedOrderPolicy(BasePolicy):
    """
    Fixed order quantity policy.
    Always orders a fixed quantity each period, regardless of state.
    """
    order_quantity: int = 30
    price_per_unit: float = 20.0

    def set_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_action(self, node_info) -> int:
        available_capacity = self.node.capacity - max(0, node_info.inventory_level)

        return int(min(self.order_quantity, max(0, available_capacity)))
