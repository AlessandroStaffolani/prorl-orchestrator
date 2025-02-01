import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Union, List, Any, Optional, Tuple

import numpy as np

from prorl.common.enum_utils import ExtendedEnum
from prorl.environment.node_groups import NodeGroups


class ActionType(str, ExtendedEnum):
    Add = 'add_node'
    Remove = 'remove_node'
    ResourceClass = 'resource_class'
    Quantity = 'quantity'
    Combined = 'combined'
    AddAction = 'add-action'
    RemoveAction = 'remove-action'


WAIT_ACTION = 'WAIT'


@dataclass
class Action:
    add_node: int
    remove_node: int
    resource_class: int
    quantity: int

    def to_dict(self):
        return {
            'add_node': self.add_node,
            'remove_node': self.remove_node,
            'resource_class': self.resource_class,
            'quantity': self.quantity
        }

    def __dict__(self):
        return self.to_dict()

    def __eq__(self, other: 'Action'):
        return self.add_node == other.add_node \
            and self.remove_node == other.remove_node \
            and self.resource_class == other.resource_class \
            and self.quantity == other.quantity

    def __iter__(self):
        for value in self.__dict__().values():
            yield value

    def __getitem__(self, item):
        if isinstance(item, int):
            return tuple(self)[item]
        else:
            return self.to_dict()[item]


class ActionSpace:

    def __init__(
            self,
            size: int,
            actions_mapping: Dict[int, Any] = None,
            add_wait_action: bool = True
    ):
        self.add_wait_action: bool = add_wait_action
        space_size = size
        self.wait_action_index: Optional[int] = None
        if self.add_wait_action:
            space_size += 1
            self.wait_action_index: int = size
        self._actions: np.ma.masked_array = np.ma.array(data=np.arange(0, space_size), mask=False)
        if actions_mapping is None:
            mapping = {a: a for a in self._actions.data}
        else:
            mapping = actions_mapping
        if self.wait_action_index is not None:
            mapping[self.wait_action_index] = WAIT_ACTION
        self.actions_mapping: Dict[int, Any] = mapping
        self.inverted_actions_mapping: Dict[Any, int] = {v: k for k, v in self.actions_mapping.items()}

    def __getitem__(self, item):
        return self._actions[item]

    def __contains__(self, item):
        return item in self._actions

    def get_available_actions(self) -> np.ndarray:
        return self._actions.compressed()

    def get_all_actions(self) -> np.ndarray:
        return self._actions.data

    def get_disabled_actions(self) -> np.ndarray:
        return self._actions.data[self._actions.mask]

    def get_mask(self) -> np.ndarray:
        return self._actions.mask

    def unmask_all(self):
        self._actions.mask = False

    def is_action_available(self, index: int):
        return not self._actions.mask[index]

    def disable_action(self, index: Union[int, slice]):
        self._actions.mask[index] = True

    def disable_wait_action(self):
        if self.add_wait_action is not None:
            self.disable_action(self.wait_action_index)

    def disable_all_except_wait(self, ensure_wait_enabled=False):
        for i, _ in enumerate(self._actions.mask):
            if i != self.wait_action_index:
                self.disable_action(i)
            else:
                if ensure_wait_enabled:
                    self.enable_action(i)

    def enable_action(self, index: Union[int, slice]):
        self._actions.mask[index] = False

    def enable_wait_action(self):
        if self.add_wait_action:
            self.enable_action(self.wait_action_index)

    def size(self, no_wait=False):
        if no_wait and self.wait_action_index is not None:
            return self._actions.size - 1
        return self._actions.size

    def shape(self):
        return self._actions.shape

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True
    ) -> Union[int, np.ndarray]:
        actions = self.get_available_actions() if compressed else self.get_all_actions()
        if random_state is None:
            return np.random.choice(actions, size=size, replace=replace, p=p)
        else:
            return random_state.choice(actions, size=size, replace=replace, p=p)

    def get_action_name(self, index: int):
        return self.actions_mapping[index]

    def is_wait_action(self, index: int) -> bool:
        return self.wait_action_index == index

    def get_action_index(self, name: str):
        return self.inverted_actions_mapping[name]

    def __str__(self):
        return f'<ActionSpace actions={self._actions} >'


class ActionSpaceWrapper:

    def __init__(
            self,
            n_nodes: int,
            n_quantities: int,
            n_resource_classes: int,
            nodes_mapping: Dict[int, Any] = None,
            quantities_mapping: Dict[int, Any] = None,
            resource_classes_mapping: Dict[int, Any] = None,
            add_wait_action: bool = True,
            node_groups: Optional[NodeGroups] = None,
            add_full_space: bool = False,
    ):
        self.add_full_space: bool = add_full_space
        if node_groups is None:
            self.add_node_space: ActionSpace = ActionSpace(size=n_nodes, actions_mapping=nodes_mapping,
                                                           add_wait_action=add_wait_action)
        else:
            actions_mapping = node_groups.get_action_space_mapping()
            self.add_node_space: ActionSpace = ActionSpace(
                size=node_groups.size(),
                actions_mapping=actions_mapping,
                add_wait_action=add_wait_action)

        self.remove_node_space: ActionSpace = ActionSpace(size=n_nodes, actions_mapping=nodes_mapping,
                                                          add_wait_action=add_wait_action)

        self.resource_classes_space: ActionSpace = ActionSpace(size=n_resource_classes,
                                                               actions_mapping=resource_classes_mapping,
                                                               add_wait_action=add_wait_action)
        self.quantity_space: ActionSpace = ActionSpace(size=n_quantities, actions_mapping=quantities_mapping,
                                                       add_wait_action=add_wait_action)
        if add_full_space:
            nodes = [i for i in range(n_nodes)]
            full_space = itertools.product(nodes, nodes)
            full_mapping = {
                i: a_nodes for i, a_nodes in enumerate(full_space)
            }
            self.full_space: ActionSpace = ActionSpace(
                size=len(full_mapping), actions_mapping=full_mapping, add_wait_action=add_wait_action
            )
        tmp = 1

    def disable_node(self, index: int):
        self.remove_node_space.disable_action(index)
        self.add_node_space.disable_action(index)
        if self.add_full_space:
            for a_index, a_nodes in self.full_space.actions_mapping.items():
                if a_nodes[0] == index or a_nodes[1] == index:
                    self.full_space.disable_action(a_index)

    def unmask_all_nodes(self):
        self.remove_node_space.unmask_all()
        self.add_node_space.unmask_all()
        if self.add_full_space:
            self.full_space.unmask_all()

    def unmask_all(self):
        self.unmask_all_nodes()
        self.resource_classes_space.unmask_all()
        self.quantity_space.unmask_all()
        if self.add_full_space:
            self.full_space.unmask_all()

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True,
    ) -> Union[Action, List[Action]]:
        a_remove = self.remove_node_space.sample(size, replace, p, random_state, compressed)
        a_add = self.add_node_space.sample(size, replace, p, random_state, compressed)
        a_res_class = self.resource_classes_space.sample(size, replace, p, random_state, compressed)
        a_quantity = self.quantity_space.sample(size, replace, p, random_state, compressed)
        if size is None:
            return Action(remove_node=a_remove, add_node=a_add, resource_class=a_res_class, quantity=a_quantity)
        else:
            action_tuple = list(zip(a_remove, a_add, a_res_class, a_quantity))
            return [Action(*values) for values in action_tuple]

    def is_wait_action(self, action: Action):
        return self.add_node_space.is_wait_action(action.add_node) \
            or self.quantity_space.is_wait_action(action.quantity) \
            or self.resource_classes_space.is_wait_action(action.resource_class) \
            or self.remove_node_space.is_wait_action(action.remove_node)

    def enable_wait_action(self):
        self.remove_node_space.enable_wait_action()
        self.add_node_space.enable_wait_action()
        self.resource_classes_space.enable_wait_action()
        self.quantity_space.enable_wait_action()
        if self.add_full_space:
            self.full_space.enable_wait_action()


class CombinedActionSpaceWrapper(ActionSpaceWrapper):

    def __init__(
            self,
            n_nodes: int,
            n_quantities: int,
            n_resource_classes: int,
            nodes_mapping: Dict[int, Any] = None,
            quantities_mapping: Dict[int, Any] = None,
            resource_classes_mapping: Dict[int, Any] = None,
            add_wait_action: bool = True,
            node_groups: Optional[NodeGroups] = None,
            add_full_space: bool = False,
    ):
        self.add_full_space: bool = add_full_space
        mapping: Dict[int, Any] = {}
        combined_mapping: Dict[int, Tuple[int, int, int]] = {}
        index = 0
        for node_i in range(n_nodes):
            for r_i, res_name in resource_classes_mapping.items():
                for q_i, quantity_val in quantities_mapping.items():
                    mapping[index] = (node_i, res_name, quantity_val)
                    combined_mapping[index] = (node_i, r_i, q_i)
                    index += 1
        super(CombinedActionSpaceWrapper, self).__init__(
            n_nodes=n_nodes,
            n_quantities=n_quantities,
            n_resource_classes=n_resource_classes,
            nodes_mapping=nodes_mapping,
            quantities_mapping=quantities_mapping,
            resource_classes_mapping=resource_classes_mapping,
            add_wait_action=add_wait_action,
            node_groups=node_groups
        )
        self.combined_space: ActionSpace = ActionSpace(
            size=n_nodes * n_resource_classes * n_quantities,
            actions_mapping=deepcopy(mapping),
            add_wait_action=add_wait_action
        )
        self.add_action_space: ActionSpace = ActionSpace(
            size=n_nodes * n_resource_classes * n_quantities,
            actions_mapping=deepcopy(mapping),
            add_wait_action=add_wait_action
        )
        self.remove_action_space: ActionSpace = ActionSpace(
            size=n_nodes * n_resource_classes * n_quantities,
            actions_mapping=deepcopy(mapping),
            add_wait_action=add_wait_action
        )
        combined_mapping[self.combined_space.wait_action_index] = (
            self.remove_node_space.wait_action_index,
            self.resource_classes_space.wait_action_index,
            self.quantity_space.wait_action_index
        )
        self.combined_mapping: Dict[int, Tuple[int, int, int]] = combined_mapping
        self.combined_inverted_mapping: Dict[Tuple[int, int, int], int] = {v: k for k, v in combined_mapping.items()}
        if add_full_space:
            nodes = [i for i in range(n_nodes)]
            if add_wait_action:
                nodes.append(WAIT_ACTION)
            full_space = itertools.product(nodes, nodes)
            full_mapping = {}
            index = 0
            for i, a_nodes in enumerate(full_space):
                if a_nodes[0] != a_nodes[1] or a_nodes[0] == WAIT_ACTION or a_nodes[1] == WAIT_ACTION:
                    full_mapping[index] = a_nodes
                    index += 1
            self.full_space: ActionSpace = ActionSpace(
                size=len(full_mapping), actions_mapping=full_mapping, add_wait_action=False
            )

    def unmask_all(self):
        self.unmask_all_nodes()
        self.combined_space.unmask_all()
        self.add_action_space.unmask_all()
        self.remove_action_space.unmask_all()
        if self.add_full_space:
            self.full_space.unmask_all()

    def is_wait_action(self, action: Action):
        return self.add_node_space.is_wait_action(action.add_node) \
            or self.quantity_space.is_wait_action(action.quantity) \
            or self.resource_classes_space.is_wait_action(action.resource_class) \
            or self.remove_node_space.is_wait_action(action.remove_node)

    def enable_wait_action(self):
        self.remove_node_space.enable_wait_action()
        self.add_node_space.enable_wait_action()
        self.resource_classes_space.enable_wait_action()
        self.combined_space.enable_wait_action()
        self.add_action_space.enable_wait_action()
        self.remove_action_space.enable_wait_action()
        if self.add_full_space:
            self.full_space.enable_wait_action()
