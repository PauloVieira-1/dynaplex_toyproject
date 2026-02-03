from __future__ import annotations

from dataclasses import dataclass
from typing import List

from node import Node
from custom_types import State

@dataclass
class BasePolicy:
    """
    Abstract base class for inventory policies.
    """
    node: Node

    def decide_order_quantity(self, pending_orders: List[int], state: State) -> int:
        raise NotImplementedError

    def set_parameters(self, **kwargs) -> None:
        raise NotImplementedError


@dataclass
class BaseStockPolicy(BasePolicy):
    """
    Base-stock policy with safety stock.
    Orders enough to reach target inventory + safety stock.
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

    def decide_order_quantity(self, pending_orders: List[int], state: State) -> int:
        total_pending = sum(pending_orders)
        inventory_position = state.inventory + total_pending - state.backorders
        base_stock_level = self.target_inventory + self.safety_stock
        order_quantity = max(0, base_stock_level - inventory_position)
        return min(order_quantity, self.node.capacity - state.inventory)



@dataclass
class MinMaxPolicy(BasePolicy):
    """
    Min-max inventory policy.
    Orders up to max_inventory when below min_inventory.
    """
    min_inventory: int = 20
    max_inventory: int = 80
    price_per_unit: float = 20.0

    def set_parameters(
        self,
        min_inventory: int | None = None,
        max_inventory: int | None = None,
        price_per_unit: float | None = None,
    ) -> None:
        if min_inventory is not None:
            self.min_inventory = min_inventory
        if max_inventory is not None:
            self.max_inventory = max_inventory
        if price_per_unit is not None:
            self.price_per_unit = price_per_unit

    def decide_order_quantity(self, pending_orders: List[int], state: State) -> int:
        total_pending = sum(pending_orders)
        inventory_position = state.inventory + total_pending

        if inventory_position >= self.max_inventory:
            return 0
        if inventory_position < self.min_inventory:
            order_quantity = self.max_inventory - inventory_position
            return min(order_quantity, self.node.capacity - state.inventory - total_pending)
        return 0


@dataclass
class FixedOrderPolicy(BasePolicy):
    """
    Fixed order quantity policy.
    Always orders a fixed quantity up to available capacity.
    """
    order_quantity: int = 30
    price_per_unit: float = 20.0

    def set_parameters(
        self,
        order_quantity: int | None = None,
        price_per_unit: float | None = None,
    ) -> None:
        if order_quantity is not None:
            self.order_quantity = order_quantity
        if price_per_unit is not None:
            self.price_per_unit = price_per_unit

    def decide_order_quantity(self, pending_orders: List[int], state: State) -> int:
        inventory_position = state.inventory + sum(pending_orders)
        available_capacity = self.node.capacity - inventory_position
        return min(self.order_quantity, max(0, available_capacity))
