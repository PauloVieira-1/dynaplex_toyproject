from custom_types import Node
from models.policy import BaseStockPolicy, FixedOrderPolicy, MinMaxPolicy, RandomChoice
import json

POLICY_CLASS_MAP = {
    "BASE_STOCK": BaseStockPolicy,
    "FIXED_ORDER": FixedOrderPolicy,
    "MIN_MAX": MinMaxPolicy,
    "FIXED_ORDER_QUANTITY": FixedOrderPolicy,
    "RANDOM": RandomChoice
}

class AssemblyTree:
    def __init__(self, json_file: str):
        
        self.assembly_tree: list[Node] = []
        self.json_file: str = json_file
        self.node_map: dict[int, Node] = {}
        self.policy_map: dict[int, object] = {}
        self.policy_list: list = [] 

    def get_policy_list(self) -> list:
        return [self.policy_map.get(node.id) for node in self.assembly_tree]

    def get_assembly_tree(self) -> list[Node]:
        return self.assembly_tree

    def set_assembly_tree(self, assembly_tree: list[Node]) -> None:
        self.assembly_tree = assembly_tree
        self.node_map = {node.id: node for node in assembly_tree}

    def get_node_by_id(self, node_id: int) -> Node | None:
        return self.node_map.get(node_id)

    def print_assembly_tree(self) -> None:
        print("Assembly Tree:")

        for node in self.assembly_tree:
            print(node)

            if node.id in self.policy_map:
                policy = self.policy_map[node.id]
                print(f"  policy -> {policy} (type: {type(policy).__name__})")
                
            if hasattr(node, "children"):
                for child in node.children:
                    print(f"  child -> {child}")

    def create_tree_from_json(self) -> None:

        with open(self.json_file) as f:
            data = json.load(f)

        print(f"Loaded {self.json_file} successfully.")

        for node_data in data["nodes"]:

            new_node = Node(
                id=node_data["id"],
                name=node_data["name"],
                capacity=node_data["capacity"],
                holding_cost=node_data["holding_cost"],
                backlog_cost=node_data["backlog_cost"],
                order_cost=node_data["order_cost"],
                lead_time=node_data["lead_time"],
                upstream_ids=node_data["upstream_ids"],
                downstream_ids=node_data["downstream_ids"],
            )

            self.assembly_tree.append(new_node)

            if "policy" in node_data:

                policy_info = node_data["policy"]
                policy_type = policy_info.get("type")

                policy_kwargs = {k: v for k, v in policy_info.items() if k != "type"}
                policy_class = POLICY_CLASS_MAP.get(policy_type)

                if not policy_class:
                    raise ValueError(f"Unknown policy type '{policy_type}' for node '{new_node.name}'")
                self.policy_map[new_node.id] = policy_class(node=new_node, **policy_kwargs)

        self.node_map = {node.id: node for node in self.assembly_tree}
        self.policy_list = [self.policy_map.get(node.id) for node in self.assembly_tree]

        print(f"Created assembly tree from {self.json_file} successfully.")