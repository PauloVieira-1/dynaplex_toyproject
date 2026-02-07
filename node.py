from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from dynaplex.modelling import (
    Features,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    discover_num_features,
)

from custom_types import State


# A single node in the supply chain, modeled as an MDP. 
# Based on Airplane in MDP examples from DynaPlex.
@dataclass(init=False, slots=True)
class Node:

    # Static configuration
    id: int
    name: str
    capacity: int
    node_type: str
    holding_cost: float
    backlog_cost: float
    order_cost: float
    lead_time: int
    upstream_ids: list[int]
    downstream_ids: list[int]

    # MDP metadata
    num_actions: int
    horizon_type: HorizonType
    num_features: int
    initial_horizon: int

    def __init__(
        self,
        id: int,
        name: str,
        capacity: int,
        node_type: str,
        holding_cost: float,
        backlog_cost: float,
        order_cost: float,
        lead_time: int,
        upstream_ids: list[int],
        downstream_ids: list[int],
        initial_horizon: int,
    ):
        assert capacity > 0, "Capacity must be positive."
        assert holding_cost >= 0, "Holding cost must be non-negative."
        assert backlog_cost >= 0, "Backlog cost must be non-negative."
        assert order_cost >= 0, "Order cost must be non-negative."
        assert lead_time >= 0, "Lead time must be non-negative."
        assert initial_horizon > 0, "Initial horizon must be positive."

        self.id = id
        self.name = name
        self.capacity = capacity
        self.node_type = node_type
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.upstream_ids = upstream_ids # Not used now for single node, but will be important for multi-node supply chain
        self.downstream_ids = downstream_ids  # Not used now for single node, but will be important for multi-node supply chain
        self.initial_horizon = initial_horizon

        # Order quantity can be from 0 to capacity (inclusive). This differs from airplane example (only reject or accept) because we have a range of order quantities.
        self.num_actions = self.capacity + 1 

        self.horizon_type = HorizonType.FINITE
        self.num_features = discover_num_features(self)

    def get_initial_state(self, context: TrajectoryContext) -> State:
        """
        Returns a fresh initial state of the node.
        """
        return State(
            inventory=self.capacity // 2, # Start with half of capacity as initial inventory (arbitrary choice for now)
            backorders=0,
            remaining_time=self.initial_horizon,
            day=0,
            pipeline=[0 for _ in range(self.lead_time)],
            category=StateCategory.AWAIT_EVENT,
        )

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        """
        Process exogenous event
        receive arrivals + realize demand.
        Moves node from AWAIT_EVENT -> AWAIT_ACTION.
        """

        # Sanity checks
        assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
        assert state.remaining_time > 0, "Simulation already finished."

        # Receive arrivals from the pipeline (lead time)
        if state.pipeline and self.lead_time > 0:
            arrived = state.pipeline.pop(0)
            state.inventory = min(self.capacity, state.inventory + arrived) # Orders that arrive but exceed capacity are lost

        # Demand realization
        demand = int(context.rng.integers(low=0, high=10)) # Comparable to "price" from dynaplex example

        fulfilled = min(demand, state.inventory)
        state.inventory -= fulfilled
        state.backorders += demand - fulfilled

        # Holding and backlog costs for the day
        context.cumulative_cost += self.holding_cost * state.inventory # Assumes is the same for each unit in inventory
        context.cumulative_cost += self.backlog_cost * state.backorders

        # Advance time
        state.day += 1
        state.remaining_time -= 1
        context.time_elapsed += 1

        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_ACTION

    def modify_state_with_action(self, state: State, context: TrajectoryContext, action: int) -> None:
        """
        Apply an action (order quantity).
        Moves node from AWAIT_ACTION -> AWAIT_EVENT or FINAL.
        This is called after the agent selects an action, and is responsible for updating the state based on that action and applying any costs associated with the action.
        """
        assert state.category == StateCategory.AWAIT_ACTION, "Not expecting an action right now."
        assert state.remaining_time > 0, "Simulation already finished."
        assert 0 <= action < self.num_actions, "Action out of bounds." 

        if action > 0:
            max_order = max(self.capacity - state.inventory, 0)
            order_qty = min(action, max_order) # Based on policies (see policy.py), action should never exceed max_order.
            context.cumulative_cost += self.order_cost * order_qty

            if self.lead_time <= 0:
                state.inventory = min(self.capacity, state.inventory + order_qty)
            else:
                if len(state.pipeline) < self.lead_time: # This should never happen because we always append lead_time zeros at the start, but just in case to avoid index errors
                    state.pipeline.extend([0] * (self.lead_time - len(state.pipeline))) # in case pipeline got shortened for some reason, we want to make sure it always has at least lead_time elements to avoid index errors
                state.pipeline[-1] += order_qty

        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_EVENT

    def write_features(self, state: State, features: Features) -> None: # Not used at the moment 
        """
        Write the features of the current state into the provided Features object.
        """
        features.append(state.inventory / self.capacity)
        features.append(state.backorders / self.capacity)
        features.append(state.remaining_time / self.initial_horizon)
        features.append(sum(state.pipeline) / self.capacity)

    def write_action_validity(self, state: State, valid: NDArray) -> None: # Not used at the moment 
        """
        Write the validity of actions for the current state into the provided valid array.
        """
        max_order = max(self.capacity - state.inventory, 0)
        for action in range(self.num_actions):
            valid[action] = action <= max_order
