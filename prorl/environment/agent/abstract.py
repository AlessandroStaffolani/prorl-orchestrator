import copy
import pickle
import time
from abc import abstractmethod
from collections import namedtuple
from logging import Logger
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch

from prorl import SingleRunConfig, single_run_config, logger
from prorl.common.data_structure import RunMode
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import Action, ActionSpaceWrapper, ActionType, CombinedActionSpaceWrapper
from prorl.environment.agent import AgentType
from prorl.environment.agent.experience_replay import ReplayBuffer, ReplayBufferType
from prorl.environment.node import Node, POOL_NODE
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import update_state_one_hot_encoding_feature, StateFeatureName, StateType, \
    FEATURE_RESOURCE_MAPPING

AgentLoss = namedtuple('AgentLoss', 'add_net remove_net resource_class_net quantity_net')
AgentQEstimate = namedtuple('AgentQEstimate', 'add_net remove_net resource_class_net quantity_net')


def _is_action_possible(nodes: List[Node], remove_node_index: int, quantity_val: int,
                        resource_class_val: str, resource: str) -> bool:
    allocated = nodes[remove_node_index].get_current_allocated(resource, res_class=resource_class_val)
    return allocated - quantity_val >= 0


def get_pool_node_index(nodes: List[Node]) -> Optional[int]:
    for i, node in enumerate(nodes):
        if node.base_station_type == POOL_NODE:
            return i
    return None


class SubActionUtils:

    def __init__(self,
                 action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper],
                 config: SingleRunConfig,
                 are_combined_actions: bool = True,
                 is_multi_reward: bool = True
                 ):
        self.config: SingleRunConfig = config
        self.use_pool_node: bool = self.config.environment.nodes.use_pool_node
        self.action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper] = action_space_wrapper
        self.are_combined_actions = are_combined_actions
        self.is_multi_reward: bool = is_multi_reward
        self.add_node_with_node_groups = self._is_add_node_with_node_groups()

    def _is_add_node_with_node_groups(self) -> bool:
        add_node_features = self.config.environment.state.base_features
        return StateFeatureName.NodeGrouped in add_node_features

    def _get_add_sub_action_state(self, state: State, resource: str) -> State:
        if self.are_combined_actions and self.is_multi_reward:
            add_node_features = self.config.environment.state.base_features
            state_features: List[str] = []
            for feature in add_node_features:
                name = feature.value
                if FEATURE_RESOURCE_MAPPING[feature]:
                    name = f'{feature.value}_{resource}'
                state_features.append(name)
        elif self.are_combined_actions:
            state_features: List[str] = [f.name for f in state.features()[: -1]]
        else:
            state_features: List[str] = [f.name for f in state.features()[: -3]]
        return State.get_state_slice(state, features=state_features)

    def _get_remove_sub_action_state(self, state: State, add_node: int) -> State:
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            update_state_one_hot_encoding_feature(state, feature_name=StateFeatureName.NodeAdd,
                                                  index=add_node, reset_value=0)
        return state

    def _get_resource_class_sub_action_state(self, state: State, add_node: int, remove_node: int) -> State:
        state_features: List[str] = [f.name for f in state.features()[: -1]]
        new_state = State.get_state_slice(state, features=state_features)
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            update_state_one_hot_encoding_feature(new_state, feature_name=StateFeatureName.NodeAdd,
                                                  index=add_node, reset_value=0)
            update_state_one_hot_encoding_feature(new_state, feature_name=StateFeatureName.NodeRemove,
                                                  index=remove_node, reset_value=0)
        return new_state

    def _get_quantity_sub_action_state(self,
                                       state: State,
                                       remove_node: int,
                                       add_node: int,
                                       resource_class: int,
                                       resource: str) -> State:
        if not self.action_space_wrapper.remove_node_space.is_wait_action(remove_node):
            update_state_one_hot_encoding_feature(state, feature_name=StateFeatureName.NodeAdd,
                                                  index=add_node, reset_value=0)
            update_state_one_hot_encoding_feature(state, feature_name=StateFeatureName.NodeRemove,
                                                  index=remove_node, reset_value=0)
            # update_state_one_hot_encoding_feature(state,
            #                                       feature_name=f'{StateFeatureName.ResourceClass.value}_{resource}',
            #                                       index=resource_class, reset_value=0)
        return state

    def _get_combined_sub_action_state(self, state: State, add_node: int, add_node_index: int):
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            update_state_one_hot_encoding_feature(state, feature_name=StateFeatureName.NodeAdd,
                                                  index=add_node_index, reset_value=0)
        return state

    def shrink_actions_based_on_node_groups(self, node_groups: Optional[NodeGroups], resource: str):
        if node_groups is not None and self.add_node_with_node_groups:
            group = node_groups.get_groups(resource_name=resource)
            for g_name, nodes in group.items():
                if len(nodes) == 0:
                    # the group need to be disabled both for add_node because no nodes belong to it
                    group_action_index = self.action_space_wrapper.add_node_space.inverted_actions_mapping[g_name]
                    self.action_space_wrapper.add_node_space.disable_action(group_action_index)

    def _shrink_after_add_node_action(self, add_node: int):
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            # if add node is wait the other can only select wait
            self.action_space_wrapper.remove_node_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.resource_classes_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.quantity_space.disable_all_except_wait(ensure_wait_enabled=True)
        else:
            # the other can not select to wait
            self.action_space_wrapper.remove_node_space.disable_wait_action()
            self.action_space_wrapper.resource_classes_space.disable_wait_action()
            self.action_space_wrapper.quantity_space.disable_wait_action()

    def _shrink_remove_node_action_space(self, add_node: int, nodes: List[Node], resource: str,
                                         node_groups: Optional[NodeGroups] = None):
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            if node_groups is None or not self.add_node_with_node_groups:
                self.action_space_wrapper.remove_node_space.disable_action(add_node)
                # check if a node has 0 resource allocated it has to be disabled
                for action in self.action_space_wrapper.remove_node_space.get_all_actions():
                    if not self.action_space_wrapper.remove_node_space.is_wait_action(action):
                        node_index = self.action_space_wrapper.remove_node_space.actions_mapping[action]
                        if node_index != add_node:
                            node = nodes[node_index]
                            if node.get_current_allocated(resource) == 0:
                                self.action_space_wrapper.remove_node_space.disable_action(action)
                            else:
                                self.action_space_wrapper.remove_node_space.enable_action(action)
                if len(self.action_space_wrapper.remove_node_space.get_available_actions()) == 0:
                    self.action_space_wrapper.remove_node_space.enable_wait_action()
                    self.action_space_wrapper.add_node_space.enable_wait_action()
                    self.action_space_wrapper.quantity_space.enable_wait_action()
                    self.action_space_wrapper.resource_classes_space.enable_wait_action()

    def _shrink_resource_class_action_space(self, add_node: int, remove_node: int, nodes: List[Node], resource: str):
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            if self.action_space_wrapper.remove_node_space.is_wait_action(remove_node):
                # if remove_node is WAIT we have to wait
                self.action_space_wrapper.resource_classes_space.enable_wait_action()
                self.action_space_wrapper.resource_classes_space.disable_all_except_wait()
            else:
                node = nodes[remove_node]
                for action in self.action_space_wrapper.resource_classes_space.get_all_actions():
                    if not self.action_space_wrapper.resource_classes_space.is_wait_action(action):
                        action_name = self.action_space_wrapper.resource_classes_space.actions_mapping[action]
                        node_units = node.get_current_resource_class_units(resource, res_class=action_name)
                        if node_units == 0:
                            # no units for this resource so we disable the action
                            self.action_space_wrapper.resource_classes_space.disable_action(action)
                        else:
                            # there is at least 1 unit of the resource class, so we enable it in case it was disabled
                            self.action_space_wrapper.resource_classes_space.enable_action(action)

    def _shrink_quantity_action_space(self, remove_node: int, resource_class: int, nodes: List[Node], resource: str,
                                      node_groups: Optional[NodeGroups] = None):
        if not self.action_space_wrapper.remove_node_space.is_wait_action(remove_node):
            if self.action_space_wrapper.resource_classes_space.is_wait_action(resource_class):
                self.action_space_wrapper.quantity_space.enable_wait_action()
                self.action_space_wrapper.quantity_space.disable_all_except_wait()
            else:
                if node_groups is None or not self.add_node_with_node_groups:
                    node = nodes[remove_node]
                    resource_class_units = node.get_current_resource_class_units(
                        resource,
                        res_class=self.action_space_wrapper.resource_classes_space.actions_mapping[resource_class])
                    for action in self.action_space_wrapper.quantity_space.get_all_actions():
                        if not self.action_space_wrapper.quantity_space.is_wait_action(action):
                            action_val = self.action_space_wrapper.quantity_space.actions_mapping[action]
                            if action_val <= resource_class_units:
                                self.action_space_wrapper.quantity_space.enable_action(action)
                            else:
                                self.action_space_wrapper.quantity_space.disable_action(action)
                    if len(self.action_space_wrapper.quantity_space.get_available_actions()) == 0:
                        # no action available, meaning that the remove_node selected has 0 resource, we have to wait
                        self.action_space_wrapper.remove_node_space.enable_wait_action()
                        self.action_space_wrapper.add_node_space.enable_wait_action()
                        self.action_space_wrapper.quantity_space.enable_wait_action()
                        self.action_space_wrapper.resource_classes_space.enable_wait_action()
        else:
            self.action_space_wrapper.quantity_space.enable_wait_action()

    def _shrink_combined_action_space(self, add_node: int, nodes: List[Node], resource: str, add_node_index: int):
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            self.action_space_wrapper.combined_space.disable_all_except_wait()
            self.action_space_wrapper.combined_space.enable_wait_action()
        else:
            self.action_space_wrapper.combined_space.unmask_all()
            self.action_space_wrapper.combined_space.disable_wait_action()
            for action in self.action_space_wrapper.combined_space.get_available_actions():
                if not self.action_space_wrapper.combined_space.is_wait_action(action):
                    remove_node, resource_class, quantity = self.action_space_wrapper.combined_space.actions_mapping[
                        action]
                    if remove_node == add_node_index:
                        self.action_space_wrapper.combined_space.disable_action(action)
                    else:
                        if _is_action_possible(nodes=nodes, resource=resource, resource_class_val=resource_class,
                                               remove_node_index=remove_node, quantity_val=quantity):
                            self.action_space_wrapper.combined_space.enable_action(action)
                        else:
                            self.action_space_wrapper.combined_space.disable_action(action)
            if len(self.action_space_wrapper.combined_space.get_available_actions()) == 0:
                self.action_space_wrapper.combined_space.enable_wait_action()

    def _shrink_add_node_with_pool(self, nodes: List[Node], resource: str):
        pool_node_index = get_pool_node_index(nodes)
        pool_node = nodes[pool_node_index]
        unused_resources = pool_node.get_current_allocated(resource)
        if unused_resources == 0:
            self.action_space_wrapper.add_node_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.add_node_space.enable_action(pool_node_index)
        else:
            all_empty = True
            for index, node in enumerate(nodes):
                if index != pool_node_index:
                    node_res = node.get_current_allocated(resource)
                    if node_res > 0:
                        all_empty = False
            if all_empty:
                self.action_space_wrapper.add_node_space.unmask_all()
                self.action_space_wrapper.add_node_space.disable_action(pool_node_index)

    def _shrink_remove_node_with_pool(self, add_node: int, nodes: List[Node], resource: str):
        pool_node_index = get_pool_node_index(nodes)
        self.action_space_wrapper.remove_node_space.unmask_all()
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            self.action_space_wrapper.remove_node_space.disable_all_except_wait(ensure_wait_enabled=True)
        else:
            self.action_space_wrapper.remove_node_space.disable_wait_action()
            if add_node == pool_node_index:
                # we are releasing regular nodes resources by sending them to the pool,
                # we disable the pool as remove node
                self.action_space_wrapper.remove_node_space.disable_action(pool_node_index)
                for index, node in enumerate(nodes):
                    if node.get_current_allocated(resource) == 0:
                        self.action_space_wrapper.remove_node_space.disable_action(index)
                if len(self.action_space_wrapper.remove_node_space.get_available_actions()) == 0:
                    self.action_space_wrapper.remove_node_space.enable_wait_action()
            else:
                # we want to add resources to a regular node, so we allow moving only from the pool node
                self.action_space_wrapper.remove_node_space.disable_all_except_wait()
                self.action_space_wrapper.remove_node_space.disable_wait_action()
                if nodes[pool_node_index].get_current_allocated(resource) > 0:
                    self.action_space_wrapper.remove_node_space.enable_action(pool_node_index)
                else:
                    self.action_space_wrapper.remove_node_space.enable_wait_action()

    def _shrink_resource_class_with_pool(self, add_node: int, remove_node: int):
        self.action_space_wrapper.resource_classes_space.unmask_all()
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node) or \
                self.action_space_wrapper.remove_node_space.is_wait_action(remove_node):
            self.action_space_wrapper.resource_classes_space.disable_all_except_wait(ensure_wait_enabled=True)
        else:
            self.action_space_wrapper.resource_classes_space.disable_wait_action()

    def _shrink_quantity_with_pool(
            self,
            add_node: int,
            remove_node: int,
            resource_class: int,
            nodes: List[Node],
            resource: str
    ):
        self.action_space_wrapper.quantity_space.unmask_all()
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node) or \
                self.action_space_wrapper.remove_node_space.is_wait_action(remove_node) or \
                self.action_space_wrapper.resource_classes_space.is_wait_action(resource_class):
            self.action_space_wrapper.quantity_space.disable_all_except_wait(ensure_wait_enabled=True)
        else:
            self.action_space_wrapper.quantity_space.disable_wait_action()
            node = nodes[remove_node]
            resource_class_units = node.get_current_resource_class_units(
                resource,
                res_class=self.action_space_wrapper.resource_classes_space.actions_mapping[resource_class])
            for action in self.action_space_wrapper.quantity_space.get_all_actions():
                if not self.action_space_wrapper.quantity_space.is_wait_action(action):
                    action_val = self.action_space_wrapper.quantity_space.actions_mapping[action]
                    if action_val <= resource_class_units:
                        self.action_space_wrapper.quantity_space.enable_action(action)
                    else:
                        self.action_space_wrapper.quantity_space.disable_action(action)
            if len(self.action_space_wrapper.quantity_space.get_available_actions()) == 0:
                self.action_space_wrapper.quantity_space.enable_wait_action()

    def _shrink_combined_with_pool(
            self,
            add_node: int,
            nodes: List[Node],
            resource: str,
            add_node_index: int
    ):
        self.action_space_wrapper.combined_space.unmask_all()
        self.action_space_wrapper.remove_node_space.unmask_all()
        self.action_space_wrapper.resource_classes_space.unmask_all()
        self.action_space_wrapper.quantity_space.unmask_all()
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node):
            self.action_space_wrapper.remove_node_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.resource_classes_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.quantity_space.disable_all_except_wait(ensure_wait_enabled=True)
            self.action_space_wrapper.combined_space.disable_all_except_wait(ensure_wait_enabled=True)
        else:
            self.action_space_wrapper.remove_node_space.disable_wait_action()
            self.action_space_wrapper.resource_classes_space.disable_wait_action()
            self.action_space_wrapper.quantity_space.disable_wait_action()
            self.action_space_wrapper.combined_space.disable_wait_action()
            pool_node_index = get_pool_node_index(nodes)
            for action in self.action_space_wrapper.combined_space.get_available_actions():
                if not self.action_space_wrapper.combined_space.is_wait_action(action):
                    remove_node, resource_class, quantity = self.action_space_wrapper.combined_space.actions_mapping[
                        action]
                    if add_node_index == pool_node_index:
                        # disable pool and check the other actions
                        if remove_node == pool_node_index:
                            self.action_space_wrapper.combined_space.disable_action(action)
                        else:
                            if _is_action_possible(nodes=nodes, resource=resource, resource_class_val=resource_class,
                                                   remove_node_index=remove_node, quantity_val=quantity):
                                self.action_space_wrapper.combined_space.enable_action(action)
                            else:
                                self.action_space_wrapper.combined_space.disable_action(action)
                    else:
                        # disable all except the pool and verify other pool's actions
                        if remove_node == add_node_index:
                            self.action_space_wrapper.combined_space.disable_action(action)
                        else:
                            if remove_node != pool_node_index:
                                self.action_space_wrapper.combined_space.disable_action(action)
                            else:
                                if _is_action_possible(nodes=nodes, resource=resource,
                                                       resource_class_val=resource_class,
                                                       remove_node_index=remove_node, quantity_val=quantity):
                                    self.action_space_wrapper.combined_space.enable_action(action)
                                else:
                                    self.action_space_wrapper.combined_space.disable_action(action)
            if len(self.action_space_wrapper.combined_space.get_available_actions()) == 0:
                self.action_space_wrapper.combined_space.enable_wait_action()

    def add_node_shrink(self, nodes: List[Node], resource: str):
        if self.use_pool_node:
            self._shrink_add_node_with_pool(nodes, resource)
        else:
            pass

    def remove_node_state(self, add_node: int, state: State, nodes: List[Node], resource: str,
                          node_groups: Optional[NodeGroups] = None) -> State:
        if self.use_pool_node:
            self._shrink_remove_node_with_pool(add_node, nodes, resource)
        else:
            self._shrink_after_add_node_action(add_node)
            self._shrink_remove_node_action_space(add_node=add_node,
                                                  nodes=nodes,
                                                  resource=resource,
                                                  node_groups=node_groups)
        return self._get_remove_sub_action_state(state, add_node=add_node)

    def resource_class_state(self, add_node: int, remove_node: int, nodes: List[Node],
                             resource: str, state: State) -> State:
        if self.use_pool_node:
            self._shrink_resource_class_with_pool(add_node, remove_node)
        else:
            self._shrink_resource_class_action_space(add_node, remove_node, nodes, resource)
        return self._get_resource_class_sub_action_state(state,
                                                         add_node=add_node,
                                                         remove_node=remove_node)

    def quantity_state(self, add_node: int, remove_node: int, resource_class: int,
                       state: State, nodes: List[Node], resource: str,
                       node_groups: Optional[NodeGroups] = None) -> State:
        if self.use_pool_node:
            self._shrink_quantity_with_pool(add_node, remove_node, resource_class, nodes, resource)
        else:
            self._shrink_quantity_action_space(remove_node, resource_class, nodes, resource, node_groups)
        return self._get_quantity_sub_action_state(state, remove_node, add_node,
                                                   resource_class=resource_class,
                                                   resource=resource)

    def combined_state(self, add_node: int, state: State, nodes: List[Node], resource: str,
                       add_node_index: int) -> State:
        if self.use_pool_node:
            self._shrink_combined_with_pool(add_node, nodes, resource, add_node_index)
        else:
            self._shrink_combined_action_space(add_node, nodes, resource, add_node_index)
        return self._get_combined_sub_action_state(state, add_node, add_node_index)

    def add_action_shrink(self, pool_node: Node, resource: str, nodes: List[Node]):
        if pool_node.get_current_allocated(resource) > 0:
            self.action_space_wrapper.add_action_space.unmask_all()
        else:
            self.action_space_wrapper.add_action_space.disable_all_except_wait(ensure_wait_enabled=True)

    def populate_node_add_feature(
            self,
            add_node_action: int,
            state: State
    ) -> State:
        return self._get_remove_sub_action_state(state, add_node_action)

    def remove_action_shrink(self, nodes: List[Node], resource: str, add_action_index: int):
        self.action_space_wrapper.remove_action_space.unmask_all()
        for action in self.action_space_wrapper.remove_action_space.get_available_actions():
            if not self.action_space_wrapper.remove_action_space.is_wait_action(action):
                node, resource_class, quantity = self.action_space_wrapper.remove_action_space.actions_mapping[action]
                if not self.action_space_wrapper.add_action_space.is_wait_action(add_action_index) \
                        and add_action_index == node:
                    self.action_space_wrapper.remove_action_space.disable_action(action)
                else:
                    if nodes[node].get_current_resource_class_units(resource, resource_class) >= quantity:
                        self.action_space_wrapper.remove_action_space.enable_action(action)
                    else:
                        self.action_space_wrapper.remove_action_space.disable_action(action)


class AgentAbstract:

    def __init__(
            self,
            action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper],
            random_state: np.random.RandomState,
            name: AgentType,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            **kwargs
    ):
        self.config: SingleRunConfig = config if config is not None else single_run_config
        self.logger: Logger = log
        self.name: AgentType = name
        self.mode: RunMode = mode
        self.random: np.random.RandomState = random_state
        self.action_spaces: Dict[ActionType, int] = action_spaces
        self.state_spaces: Dict[StateType, int] = state_spaces
        self.collect_rollouts: bool = False
        self.is_baseline: bool = False
        self.action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper] = action_space_wrapper
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.is_prioritized_buffer = False
        if self.config.environment.agent.replay_buffer.type == ReplayBufferType.PrioritizedBuffer:
            self.is_prioritized_buffer = True
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.combined_sub_actions: bool = self.config.environment.agent.combine_last_sub_actions
        self.is_multi_reward: bool = self.config.environment.reward.multi_reward
        self.requires_validation: bool = False
        self.save_agent_state: bool = False
        self.selected_actions: Dict[ActionType, Dict[int, int]] = {
            ActionType.Add: {},
            ActionType.Remove: {},
            ActionType.ResourceClass: {},
            ActionType.Quantity: {}
        }
        self.stats_loaded: bool = False
        self.selected_actions_loaded: Dict[ActionType, Dict[int, int]] = {
            ActionType.Add: {},
            ActionType.Remove: {},
            ActionType.ResourceClass: {},
            ActionType.Quantity: {}
        }
        self.stats_tracker: Optional[Tracker] = None
        self.current_add_node = None
        self.current_remove_node = None
        self.current_resource_class = None
        self.current_quantity = None
        self.sub_actions_utils: SubActionUtils = SubActionUtils(
            action_space_wrapper=self.action_space_wrapper,
            config=self.config,
            are_combined_actions=self.combined_sub_actions,
            is_multi_reward=self.is_multi_reward
        )
        self.sanitize_counter = 0

        self.max_cost: Dict[str, float] = {}
        self.remaining_gap_max_factor: Dict[str, float] = {}

        self.choose_action_calls: int = 0
        self.learn_calls: int = 0
        self.bootstrap_steps: int = 0

        self._init_selected_actions()
        self.add_node_with_node_groups: bool = self._is_add_node_with_node_groups()
        # self._init_replay_buffer()

    def __str__(self):
        return f'<Agent name={self.name.value} >'

    @property
    def is_bootstrapping(self) -> bool:
        return False

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker

    def set_max_factors(self, max_cost: Dict[str, float], max_remaining_gap: Dict[str, float]):
        self.max_cost = max_cost
        self.remaining_gap_max_factor = max_remaining_gap

    def _is_add_node_with_node_groups(self) -> bool:
        add_node_features = self.config.environment.state.base_features
        return StateFeatureName.NodeGrouped in add_node_features

    def _init_selected_actions(self):
        self.selected_actions[ActionType.Add] = {
            a: 0 for a, _ in self.action_space_wrapper.add_node_space.actions_mapping.items()}
        self.selected_actions[ActionType.Remove] = {
            a: 0 for a, _ in self.action_space_wrapper.remove_node_space.actions_mapping.items()}
        self.selected_actions[ActionType.ResourceClass] = {
            a: 0 for a, _ in self.action_space_wrapper.resource_classes_space.actions_mapping.items()}
        self.selected_actions[ActionType.Quantity] = {
            a: 0 for a, _ in self.action_space_wrapper.quantity_space.actions_mapping.items()}

    def get_action_epsilon(self) -> float:
        return 0

    def _update_selected_actions(self, actions: List[Action]):
        # TODO fix this
        pass
        # self.selected_actions[ActionType.Add][action.add_node] += 1
        # self.selected_actions[ActionType.Remove][action.remove_node] += 1
        # self.selected_actions[ActionType.ResourceClass][action.resource_class] += 1
        # self.selected_actions[ActionType.Quantity][action.quantity] += 1

    def _choose_with_combined(
            self,
            state_wrapper: Dict[StateType, State],
            nodes: List[Node],
            resource: str,
            epsilon: float,
            random: float,
            node_groups: Optional[NodeGroups] = None,
            demand: Optional[StepData] = None,
    ) -> Tuple[List[Action], float, Dict[StateType, State]]:
        action_time = 0
        add_node_state: State = state_wrapper[StateType.Add]
        state: State = state_wrapper[StateType.Combined]
        self.sub_actions_utils.shrink_actions_based_on_node_groups(node_groups, resource)
        # get add node sub action
        # add_sub_action_state = self.sub_actions_utils.add_node_state(add_node_state, resource)
        start = time.time()
        add_node = self._choose_node_add(add_node_state, epsilon, random, resource, nodes, node_groups, demand)
        action_time += (time.time() - start)
        if self.add_node_with_node_groups:
            add_node_index = None
            if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
                add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node]
                add_node_index = node_groups.sample_group(group_name=add_node_group, resource_name=resource)
        else:
            add_node_index = add_node

        # get combined sub action
        final_state = self.sub_actions_utils.combined_state(add_node, state, nodes, resource,
                                                            add_node_index)

        start = time.time()
        remove_node, resource_class, quantity = self._choose_combined_sub_action(final_state,
                                                                                 add_node_index,
                                                                                 nodes, epsilon, random, resource,
                                                                                 demand)
        action_time += (time.time() - start)

        # build the action tuple
        action = Action(
            remove_node=remove_node,
            add_node=add_node,
            resource_class=resource_class,
            quantity=quantity
        )
        final_state_wrapper = {
            StateType.Add: add_node_state,
            StateType.Combined: final_state
        }
        return [action], action_time, final_state_wrapper

    def _action_from_sub_action_index(self, index: int, is_add: bool) -> Action:
        node, resource_class, quantity = self.action_space_wrapper.combined_mapping[index]
        if is_add:
            return Action(
                add_node=node,
                remove_node=-1,  # -1 means the pool
                resource_class=resource_class,
                quantity=quantity
            )
        else:
            return Action(
                add_node=-1,  # -1 means the pool
                remove_node=node,
                resource_class=resource_class,
                quantity=quantity
            )

    def _choose_pool_action(
            self,
            state_wrapper: Dict[StateType, State],
            nodes: List[Node],
            resource: str,
            epsilon: float,
            random: float,
            pool_node: Node,
            node_groups: Optional[NodeGroups] = None,
            demand: Optional[StepData] = None,
    ) -> Tuple[List[Action], float, Dict[StateType, State]]:
        action_time = 0
        actions: List[Action] = []
        add_state: State = state_wrapper[StateType.Add]
        remove_state: State = state_wrapper[StateType.Remove]

        # add action
        self.sub_actions_utils.add_action_shrink(pool_node, resource, nodes)
        start = time.time()
        add_action_index = self._choose_add(add_state, epsilon, random, resource, nodes, node_groups, demand, pool_node)
        action_time += (time.time() - start)
        actions.append(self._action_from_sub_action_index(add_action_index, True))

        # remove action
        self.sub_actions_utils.remove_action_shrink(nodes, resource, add_action_index)
        if len(add_state) != len(remove_state):
            remove_state = self.sub_actions_utils.populate_node_add_feature(add_action_index, remove_state)
        start = time.time()
        remove_action_index = self._choose_remove(remove_state, epsilon, random, resource, nodes,
                                                  node_groups, demand, pool_node)
        action_time += (time.time() - start)
        actions.append(self._action_from_sub_action_index(remove_action_index, False))

        state_wrapper[StateType.Remove] = remove_state

        return actions, action_time, state_wrapper,

    def _choose_full_split(
            self,
            state_wrapper: Dict[StateType, State],
            nodes: List[Node],
            resource: str,
            epsilon: float,
            random: float,
            node_groups: Optional[NodeGroups] = None,
            demand: Optional[StepData] = None,
    ) -> Tuple[List[Action], float, Dict[StateType, State]]:
        action_time = 0
        add_node_state: State = state_wrapper[StateType.Add]
        remove_node_state: State = state_wrapper[StateType.Remove]
        quantity_state: State = state_wrapper[StateType.Quantity]
        self.sub_actions_utils.shrink_actions_based_on_node_groups(node_groups, resource)

        start = time.time()
        self.sub_actions_utils.add_node_shrink(nodes, resource)
        add_node = self._choose_node_add(add_node_state, epsilon, random, resource, nodes, node_groups, demand)
        action_time += (time.time() - start)
        if self.add_node_with_node_groups:
            add_node_index = None
            if not self.action_space_wrapper.add_node_space.is_wait_action(add_node):
                add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node]
                add_node_index = node_groups.sample_group(group_name=add_node_group, resource_name=resource)
        else:
            add_node_index = add_node

        # shrink remove node action space and build the final version of remove node state
        remove_node_state = self.sub_actions_utils.remove_node_state(
            add_node_index, remove_node_state, nodes, resource, node_groups)
        start = time.time()
        remove_node = self._choose_node_remove(remove_node_state, epsilon, random, resource, nodes, node_groups, demand)
        action_time += (time.time() - start)

        # resource class is always the same
        resource_class = self._choose_resource_class(add_node_index, remove_node)

        # shrink the quantity action space and build the final version of quantity state
        quantity_state = self.sub_actions_utils.quantity_state(
            add_node_index, remove_node, resource_class, quantity_state, nodes, resource, node_groups)

        start = time.time()
        quantity = self._choose_quantity(quantity_state, epsilon, random, resource, nodes, node_groups, demand)
        action_time += (time.time() - start)

        # build the action tuple
        action = Action(
            remove_node=remove_node,
            add_node=add_node,
            resource_class=resource_class,
            quantity=quantity
        )
        final_state_wrapper = {
            StateType.Add: add_node_state,
            StateType.Remove: remove_node_state,
            StateType.Quantity: quantity_state
        }
        return [action], action_time, final_state_wrapper

    def choose(
            self,
            state_wrapper: Dict[StateType, State],
            nodes: List[Node],
            resource: str,
            pool_node: Optional[Node] = None,
            node_groups: Optional[NodeGroups] = None,
            demand: Optional[StepData] = None,
    ) -> Tuple[List[Action], Dict[StateType, State], Optional[Dict[str, Union[str, int, float, bool]]]]:
        # for epsilon greedy only
        epsilon = self.get_action_epsilon()
        if self.mode != RunMode.Train or self.collect_rollouts:
            epsilon = 0
        random = self.random.random()

        if self.config.environment.nodes.use_pool_node:
            actions, action_time, final_state_wrapper = self._choose_pool_action(state_wrapper, nodes, resource,
                                                                                 epsilon, random, pool_node,
                                                                                 node_groups, demand)
        elif self.config.environment.sub_agents_setup.full_action_split:
            actions, action_time, final_state_wrapper = self._choose_full_split(state_wrapper, nodes, resource,
                                                                                epsilon, random, node_groups, demand)
        else:
            # use combined movement sub-agent
            actions, action_time, final_state_wrapper = self._choose_with_combined(state_wrapper, nodes, resource,
                                                                                   epsilon, random, node_groups, demand)
        # sanitize action in case we waited when add node was not waiting
        actions = self.sanitize_action(actions)
        # update action selection stats
        self._update_selected_actions(actions)
        # return the action and the last state update
        action_info: Dict[str, Union[str, int, float, bool]] = {
            'epsilon': epsilon,
            'random_action': random < epsilon,
            'wait_movement': False,
            'action_time': action_time
        }
        self._update_choose_action_stats(epsilon, random)
        self.choose_action_calls += 1
        return actions, final_state_wrapper, action_info

    def sanitize_action(self, actions: List[Action]) -> List[Action]:
        for i, action in enumerate(actions):
            if self.action_space_wrapper.is_wait_action(action):
                if not self.action_space_wrapper.add_node_space.is_wait_action(action.add_node) \
                        and action.add_node != -1 and action.remove_node != -1:
                    self.sanitize_counter += 1
                    if self.sanitize_counter == 200:
                        self.logger.warning('Action has been sanitized 200 times, resetting counter for now')
                        self.sanitize_counter = 0
                self.action_space_wrapper.add_node_space.enable_wait_action()
                self.action_space_wrapper.remove_node_space.enable_wait_action()
                self.action_space_wrapper.resource_classes_space.enable_wait_action()
                self.action_space_wrapper.quantity_space.enable_wait_action()
                actions[i] = Action(
                    remove_node=self.action_space_wrapper.remove_node_space.wait_action_index,
                    add_node=self.action_space_wrapper.add_node_space.wait_action_index,
                    resource_class=self.action_space_wrapper.resource_classes_space.wait_action_index,
                    quantity=self.action_space_wrapper.quantity_space.wait_action_index
                )
        return actions

    def _update_choose_action_stats(self, epsilon: float, random: float):
        if self.mode == RunMode.Train:
            step = self.choose_action_calls
            self.stats_tracker.track(f'{RunMode.Train.value}/action_history/is_random', random < epsilon, step)
            self.stats_tracker.track(f'hp/epsilon', epsilon, step)
            beta = self.get_prioritized_beta_parameter()
            if beta is not None:
                self.stats_tracker.track(f'hp/prioritized_buffer_beta', beta, step)
            theta = self.get_expert_train_theta()
            if theta is not None:
                self.stats_tracker.track('hp/expert_action_theta', theta, step)

    @abstractmethod
    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None) -> int:
        pass

    @abstractmethod
    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        pass

    @abstractmethod
    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        pass

    @abstractmethod
    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        pass

    @abstractmethod
    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        pass

    def _handle_combined_action(self, combined_action: int) -> Tuple[int, int, int]:
        remove_node, resource_class, quantity = self.action_space_wrapper.combined_mapping[combined_action]
        self.current_remove_node = remove_node
        self.current_resource_class = resource_class
        self.current_quantity = quantity
        return remove_node, resource_class, quantity

    def _choose_resource_class(self, add_node: int, remove_node: int) -> int:
        if self.config.environment.sub_agents_setup.full_action_split:
            self.action_space_wrapper.resource_classes_space.unmask_all()
            if self.action_space_wrapper.add_node_space.is_wait_action(add_node) \
                    or self.action_space_wrapper.remove_node_space.is_wait_action(remove_node):
                return self.action_space_wrapper.resource_classes_space.wait_action_index
            else:
                return 0
        else:
            return self.current_resource_class

    @abstractmethod
    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None) -> int:
        pass

    def learn(self, **kwargs) -> Tuple[Optional[Union[AgentLoss, float]], Optional[Union[AgentQEstimate, float]]]:
        return self._learn()

    @abstractmethod
    def _learn(self) -> Tuple[Optional[Union[AgentLoss, float]], Optional[Union[AgentQEstimate, float]]]:
        pass

    @abstractmethod
    def push_experience(
            self,
            state_wrapper: Union[Dict[str, State], np.ndarray],
            actions: Union[List[Action], np.ndarray],
            next_state_wrapper: Union[Dict[str, State], np.ndarray],
            reward: Union[float, Tuple[float, float], np.ndarray],
            done: bool
    ):
        pass

    def get_model(self, net_type: StateType = None, policy_net=True) -> Optional[torch.nn.Module]:
        return None

    def get_agent_state(self) -> Dict[str, Any]:
        return {
            'selected_actions': self.selected_actions,
        }

    def load_agent_state(self, agent_state: Dict[str, Any]):
        self.selected_actions_loaded = copy.deepcopy(agent_state['selected_actions'])
        self.stats_loaded = True
        self.selected_actions = agent_state['selected_actions']

    def set_mode(self, mode: RunMode):
        self.mode: RunMode = mode

    def serialize_agent_state(self):
        return pickle.dumps(self.get_agent_state())

    def load_serialized_agent_state(self, serialized_agent_state: bytes):
        deserialized = pickle.loads(serialized_agent_state)
        self.load_agent_state(deserialized)

    def get_prioritized_beta_parameter(self) -> Optional[float]:
        return None

    def get_expert_train_theta(self) -> Optional[float]:
        return None

    def reset(self):
        pass
