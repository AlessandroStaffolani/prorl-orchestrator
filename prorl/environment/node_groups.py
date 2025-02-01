import math
from typing import Dict, List, Optional

import numpy as np
from numpy.random import RandomState

from prorl.core.step_data import StepData
from prorl.environment.data_structure import EnvResource
from prorl.environment.node import Node


class NodeGroups:

    def __init__(
            self,
            quantity_actions: List[int],
            resources: List[EnvResource],
            random_state: Optional[RandomState] = None,
            node_groups_with_resource_classes: bool = True
    ):
        self.node_groups_with_resource_classes: bool = node_groups_with_resource_classes
        self.quantity_actions: List[int] = quantity_actions
        self.random: RandomState = random_state if random_state is not None else np.random.RandomState()
        self.resources: List[EnvResource] = resources
        self.groups: Dict[str, Dict[str, List[int]]] = {res.name: {} for res in self.resources}
        self.groups_move_mapping: Dict[str, Dict[int, str]] = {res.name: {} for res in self.resources}
        self.groups_value_mapping: Dict[str, Dict[str, int]] = {res.name: {} for res in self.resources}
        self._zero_key = 'g_0'
        self._create_groups()

    def _create_groups(self):
        for resource in self.resources:
            resource_name = resource.name
            resource_classes = resource.classes
            group_keys = []
            for i in range(2):
                # first outer loop for negative keys, second for positive ones
                for quantity in self.quantity_actions:
                    for _, res_class in resource_classes.items():
                        if self.node_groups_with_resource_classes:
                            capacity = res_class['capacity']
                        else:
                            capacity = 1
                        key = quantity * capacity
                        if i == 0:
                            key = -key
                        if key not in group_keys:
                            group_keys.append(key)
                # key zero must be in the middle
            group_keys.append(0)
            group_keys = sorted(group_keys)
            for key in group_keys:
                group_key = f'g_{key}'
                if key == 0:
                    group_key = self._zero_key
                self.groups[resource_name][group_key] = []
                self.groups_move_mapping[resource_name][key] = group_key
                self.groups_value_mapping[resource_name][group_key] = key

    def _find_group_negative_delta(self, delta: float, node_index: int, resource_name: str):
        group_keys = list(self.groups_move_mapping[resource_name].keys())
        group_keys.sort()
        for key_val in group_keys:
            if delta <= key_val:
                group_name = self.groups_move_mapping[resource_name][key_val]
                self._add_to_group(node_index, group_name, resource_name)
                return

    def _find_group_positive_delta(self, delta: float, node_index: int, resource_name: str):
        group_keys = list(self.groups_move_mapping[resource_name].keys())
        group_keys.sort(reverse=True)
        for key_val in group_keys:
            if delta >= key_val:
                group_name = self.groups_move_mapping[resource_name][key_val]
                self._add_to_group(node_index, group_name, resource_name)
                return

    def _add_to_group(self, node_index: int, group_name: str, resource: str):
        if node_index not in self.groups[resource][group_name]:
            self.groups[resource][group_name].append(node_index)

    def add_node_to_group(self, node_index: int, node: Node, step_demand: StepData, resource_name: str):
        node_demand = step_demand[resource_name][node_index][node.base_station_type]
        node_allocated = node.get_current_allocated(resource_name)
        delta = math.floor(node_allocated - node_demand)
        if delta == 0:
            self._add_to_group(node_index, self._zero_key, resource_name)
            return
        if delta > 0:
            self._find_group_positive_delta(delta=delta, node_index=node_index, resource_name=resource_name)
            return
        else:
            self._find_group_negative_delta(delta=delta, node_index=node_index, resource_name=resource_name)
            return

    def add_all_nodes(self, nodes: List[Node], current_demand: StepData):
        for resource in self.resources:
            for i, node in enumerate(nodes):
                self.add_node_to_group(node_index=i, node=node, step_demand=current_demand, resource_name=resource.name)

    def sample_group(self, group_name: str, resource_name, remove=False, random=False) -> int:
        if random:
            node_index = self.random.choice(self.groups[resource_name][group_name])
        else:
            node_index = self.groups[resource_name][group_name][0]
        if remove:
            self.groups[resource_name][group_name].remove(node_index)
        return node_index

    def get_groups(self, resource_name: Optional[str] = None):
        if len(self.resources) == 1:
            return self.groups[self.resources[0].name]
        else:
            if resource_name is None:
                raise AttributeError('resource_name is None and available resource are more than 1')
            return self.groups[resource_name]

    def get_group_by_name(self, group_name: str, resource_name: str) -> List[int]:
        return self.groups[resource_name][group_name]

    def reset(self):
        for r_name, group in self.groups.items():
            for g_name, _ in group.items():
                self.groups[r_name][g_name] = []

    def size(self) -> int:
        res_name = self.resources[0].name
        return len(self.groups[res_name])

    def get_action_space_mapping(self) -> Dict[int, str]:
        res_name = self.resources[0].name
        group = self.get_groups(res_name)
        mapping = {}
        i = 0
        for g_name, _ in group.items():
            mapping[i] = g_name
            i += 1
        return mapping

    def get_positive_groups(self, resource_name: str, include_zero_group: bool = True) -> List[int]:
        include_zero = 1 if include_zero_group is True else 0
        group_keys = list(self.groups_move_mapping[resource_name].keys())
        group_keys.sort(reverse=True)
        return group_keys[0: len(group_keys) // 2 + include_zero]

    def get_negative_groups(self, resource_name: str, include_zero_group: bool = True) -> List[int]:
        include_zero = 1 if include_zero_group is True else 0
        group_keys = list(self.groups_move_mapping[resource_name].keys())
        group_keys.sort()
        return group_keys[0: len(group_keys) // 2 + include_zero]
