"""
Simple Supply Chain MDP - Problem Definition.

This module contains the MDP class, state representation, actions, and policies
for a single-echelon inventory management problem.

Includes:
- SupplyChainMDP: The MDP class modeling a single-echelon inventory system
- State: The state class
- ReorderAction, GraphAction: Action representations
- SCGlobalState, ActionSet: Graph-based representations
- GreedyPolicy: A policy that orders when inventory falls below a threshold
- OrderUpToPolicy: A policy that always orders to reach a target inventory level
- Helper functions for action sets and cost computation
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dynaplex_playgroud.data.action_base import Action, GlobalState
from dynaplex.modelling import (
    Features,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    discover_num_features,
)


# =============================================================================
# State and Action Definitions
# =============================================================================

@dataclass(slots=True)
class State:
    """
    State representation for the supply chain MDP.
    """
    inventory_level: int  # Current inventory at the warehouse
    incoming_order: int  # Demand (order) to be fulfilled
    # This member must always be defined on any dynaplex MDP state:
    category: StateCategory = StateCategory.AWAIT_EVENT


@dataclass(slots=True)
class ReorderAction(Action):
    """
    Action representation for the supply chain MDP.
    """
    order_quantity: int  # Number of units to order


@dataclass(slots=True)
class SCGlobalState(GlobalState):
    """Global state node with normalized features."""
    inventory_level: float
    incoming_order: float
    category: StateCategory


@dataclass(slots=True)
class GraphAction(Action):
    """Graph-only action without embedded state."""
    order_quantity: int


@dataclass(slots=True)
class ActionSet:
    """Root container for global state and candidate actions."""
    global_state: SCGlobalState
    actions: list[GraphAction]


# =============================================================================
# MDP Definition
# =============================================================================

@dataclass(init=False, slots=True)
class SupplyChainMDP:
    """
    Simple Supply Chain MDP is an infinite horizon inventory management problem.

    In this problem, demands arrive stochastically and must be fulfilled from inventory.
    The decision maker can order additional inventory at a fixed cost per unit.
    Unfulfilled demand incurs a stockout cost.
    Excess inventory incurs a holding cost.
    The goal is to minimize total costs across the episode.

    Actions:
        0 to max_order_quantity: Number of units to order
    """
    max_order_quantity: int  # Maximum units that can be ordered per period
    order_cost_per_unit: float  # Cost to order one unit
    holding_cost_per_unit: float  # Cost to hold one unit of inventory per period
    stockout_cost: float  # Cost per unit of unmet demand
    demands: NDArray[np.int64]  # Possible demand values
    demand_probs: NDArray[np.float64]  # Probability of each demand
    num_actions: int
    horizon_type: HorizonType
    num_features: int

    def __init__(
            self,
            max_order_quantity: int,
            order_cost_per_unit: float,
            holding_cost_per_unit: float,
            stockout_cost: float,
            demand_probs: NDArray[np.float64],
            demands: NDArray[np.int64] | None = None,
    ):
        """
        Initialize the Supply Chain MDP with validation.

        Args:
            max_order_quantity: Maximum units that can be ordered per period (must be > 0)
            order_cost_per_unit: Cost per unit ordered (must be >= 0)
            holding_cost_per_unit: Cost per unit held in inventory per period (must be >= 0)
            stockout_cost: Cost per unit of unmet demand (must be > 0)
            demand_probs: 1D array - probability distribution over demands (must sum to 1.0)
            demands: 1D array - possible demand values (all must be >= 0)
                     If None, defaults to [0, 1, 2, ..., len(demand_probs)-1]

        Raises:
            AssertionError: If any validation checks fail
        """
        # Validating parameters
        assert max_order_quantity > 0, "max_order_quantity must be positive"
        assert order_cost_per_unit >= 0, "order_cost_per_unit must be non-negative"
        assert holding_cost_per_unit >= 0, "holding_cost_per_unit must be non-negative"
        assert stockout_cost > 0, "stockout_cost must be positive"
        assert len(demand_probs) > 0, "demand_probs must not be empty"
        assert demand_probs.ndim == 1, "demand_probs must be 1-dimensional"
        assert np.all(demand_probs >= 0), "all probabilities must be non-negative"
        assert np.isclose(np.sum(demand_probs), 1.0, atol=1e-6), "probabilities must sum to 1.0"

        # Handle demands
        if demands is None:
            demands = np.arange(len(demand_probs), dtype=np.int64)
        else:
            assert demands.ndim == 1, "demands must be 1-dimensional"
            assert len(demands) == len(demand_probs), "demands must match demand_probs length"
            assert np.all(demands >= 0), "all demands must be non-negative"

        # Set attributes
        self.max_order_quantity = max_order_quantity
        self.order_cost_per_unit = order_cost_per_unit
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_cost = stockout_cost
        self.demands = demands
        self.demand_probs = demand_probs
        # Number of actions in the MDP (0 to max_order_quantity)
        self.num_actions = max_order_quantity + 1
        # Horizon type for this MDP
        self.horizon_type = HorizonType.INFINITE
        # Automatically discover the number of features; should call last!
        self.num_features = discover_num_features(self)

    def get_initial_state(self, context: TrajectoryContext) -> State:
        """
        Generates and returns an initial state of the MDP.
        """
        return State(
            inventory_level=self.max_order_quantity,  # Start with reasonable inventory
            incoming_order=0,
            category=StateCategory.AWAIT_EVENT,
        )

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        """
        Generate a demand arrival event and modify state in place.

        Args:
            state: Current state (modified in place)
            context: Trajectory context containing rng and cumulative_cost
        """
        # Sample a demand from the distribution
        state.incoming_order = context.rng.choice(
            self.demands,
            p=self.demand_probs,
        )

        # Next, the agent must decide how many units to order
        state.category = StateCategory.AWAIT_ACTION
        # time elapsed increases by 1
        context.time_elapsed += 1

    def modify_state_with_action(self, state: State, context: TrajectoryContext, action: int) -> None:
        """
        Apply an action to the state (modify in place).

        Args:
            state: Current state (modified in place)
            context: Trajectory context containing cumulative_cost (updated in place)
            action: Number of units to order (0 to max_order_quantity)
        """
        assert 0 <= action <= self.max_order_quantity, f"Invalid action: {action}"

        # Order cost (ordering units is instantaneous in this model)
        order_cost = action * self.order_cost_per_unit
        context.cumulative_cost += order_cost

        # Add ordered units to inventory
        state.inventory_level += action

        # Fulfill demand from inventory
        if state.inventory_level >= state.incoming_order:
            # Demand can be fully met
            state.inventory_level -= state.incoming_order
            unmet_demand = 0
        else:
            # Stockout: not enough inventory
            unmet_demand = state.incoming_order - state.inventory_level
            state.inventory_level = 0

        # Incur stockout cost for unmet demand
        stockout_cost = unmet_demand * self.stockout_cost
        context.cumulative_cost += stockout_cost

        # Incur holding cost for remaining inventory
        holding_cost = state.inventory_level * self.holding_cost_per_unit
        context.cumulative_cost += holding_cost

        # Reset incoming order
        state.incoming_order = 0

        # Transition back to await next event (infinite horizon - never reaches FINAL)
        state.category = StateCategory.AWAIT_EVENT

    def write_features(self, state: State, features: Features) -> None:
        """
        Write feature vector representation of the state.

        Args:
            state: Current state to extract features from
            features: Features sink to write features to
        """
        features.append(state.inventory_level)
        features.append(state.incoming_order)

    def write_action_validity(self, state: State, valid: NDArray[np.bool_]) -> None:
        """
        Write action validity: valid[i] = True if action i is allowed in the current state
                               valid[i] = False otherwise.

        Args:
            state: Current state
            valid: Boolean array of length num_actions to write the validity mask to
        """
        # All order quantities are always valid actions in this problem:
        pass


# =============================================================================
# Policies
# =============================================================================

@dataclass(slots=True)
class GreedyPolicy:
    """
    Greedy threshold policy for the supply chain MDP.

    This policy orders to reach a target inventory level if current inventory falls below a threshold.
    """
    mdp: SupplyChainMDP
    reorder_point: int  # Order if inventory falls below this level
    target_level: int  # Order up to this level

    def get_action(self, state: State) -> int:
        """
        Orders to reach target level if inventory is below reorder point.

        Args:
            state: Current state

        Returns:
            Action (number of units to order)
        """
        if state.inventory_level < self.reorder_point:
            # Order up to target level
            order_quantity = self.target_level - state.inventory_level
            # Ensure we don't order more than max allowed
            return min(order_quantity, self.mdp.max_order_quantity)
        else:
            # No order
            return 0


@dataclass(slots=True)
class OrderUpToPolicy:
    """
    Order-up-to policy for the supply chain MDP.

    This policy always tries to order to reach a fixed target inventory level.
    """
    mdp: SupplyChainMDP
    target_level: int  # Always order to reach this level

    def get_action(self, state: State) -> int:
        """
        Always orders to reach the target inventory level.

        Args:
            state: Current state

        Returns:
            Action (number of units to order)
        """
        if state.inventory_level < self.target_level:
            order_quantity = self.target_level - state.inventory_level
            # Ensure we don't order more than max allowed
            return min(order_quantity, self.mdp.max_order_quantity)
        else:
            # No order needed
            return 0


# =============================================================================
# Helper Functions
# =============================================================================

def one_step_cost(
        inventory_level: int,
        incoming_order: int,
        action: int,
        order_cost_per_unit: float,
        holding_cost_per_unit: float,
        stockout_cost: float,
) -> float:
    """Compute the immediate cost for a single step without mutating state."""
    cost = action * order_cost_per_unit
    inventory_level += action

    if inventory_level >= incoming_order:
        inventory_level -= incoming_order
        unmet_demand = 0
    else:
        unmet_demand = incoming_order - inventory_level
        inventory_level = 0

    cost += unmet_demand * stockout_cost
    cost += inventory_level * holding_cost_per_unit
    return float(cost)


def build_action_set(
        state: State,
        mdp: SupplyChainMDP,
        reorder_actions: list[ReorderAction],
        max_demand: int,
) -> ActionSet:
    """Create an action set with a global state node and action nodes.

    Args:
        state: Current state
        mdp: The SupplyChainMDP instance
        reorder_actions: List of available ReorderAction objects
        max_demand: Maximum possible demand (for normalization)

    Returns:
        ActionSet containing global state and valid graph actions
    """
    max_inventory = mdp.max_order_quantity
    inv = float(state.inventory_level) / float(max_inventory) if max_inventory > 0 else 0.0
    dem = float(state.incoming_order) / float(max_demand) if max_demand > 0 else 0.0
    global_state = SCGlobalState(
        inventory_level=inv,
        incoming_order=dem,
        category=state.category,
    )
    # State-dependent action availability: cap order quantity by remaining capacity.
    remaining_capacity = max(0, max_inventory - state.inventory_level)
    graph_actions = [
        GraphAction(order_quantity=action.order_quantity)
        for action in reorder_actions
        if 0 <= action.order_quantity <= remaining_capacity
    ]
    if not graph_actions:
        graph_actions = [GraphAction(order_quantity=0)]
    return ActionSet(global_state=global_state, actions=graph_actions)

