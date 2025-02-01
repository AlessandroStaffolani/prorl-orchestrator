import math
from itertools import product
from logging import Logger
from typing import Optional, Dict, Tuple, List

import numpy as np

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.common.math_util import normalize_scalar, l1_distance
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, ActionType, Action, ActionSpace
from prorl.environment.agent import AgentType
from prorl.environment.agent.baseline.random import BaselineAgent
from prorl.environment.agent.policy import local_optimal_add_node_policy, local_optimal_movement_policy, \
    new_movement_local_optimal_policy, new_full_action_greedy_policy, local_optimal_add_node_policy_with_pool, \
    movement_local_optimal_policy_with_pool, local_add_or_remove_action_heuristic
from prorl.environment.data_structure import ResourceClass, EnvResource
from prorl.environment.node import Node, POOL_NODE
from prorl.environment.node_groups import NodeGroups
from prorl.environment.reward import RewardFunctionType
from prorl.environment.scalarization_function import scalarize_rewards, ScalarizationFunction
from prorl.environment.state_builder import StateType


class GreedyOptimalAgent(BaselineAgent):

    def __init__(
            self,
            action_space_wrapper: ActionSpaceWrapper,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            **kwargs
    ):
        super(GreedyOptimalAgent, self).__init__(action_space_wrapper,
                                                 random_state=random_state,
                                                 name=AgentType.Heuristic,
                                                 action_spaces=action_spaces,
                                                 state_spaces=state_spaces,
                                                 log=log,
                                                 config=config,
                                                 mode=mode,
                                                 **kwargs)
        self.n_nodes = self.config.environment.nodes.get_n_nodes()
        self.n_resource_classes = len(self.config.environment.resources[0].classes)
        self.resource_classes: Dict[str, ResourceClass] = self.config.environment.resources[0].classes
        self.quantity_movements: List[int] = self.config.environment.action_space.bucket_move
        self.is_conservative = True
        self.current_add_node: Optional[int] = None
        self.current_combined: Optional[int] = None
        self.use_pool_node: bool = self.config.environment.nodes.use_pool_node

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        return local_add_or_remove_action_heuristic(
            nodes, pool_node, resource, demand,
            resources=self.config.environment.get_env_resources(), config=self.config, is_add_action=True,
            action_space_wrapper=self.action_space_wrapper
        )

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        return local_add_or_remove_action_heuristic(
            nodes, pool_node, resource, demand,
            resources=self.config.environment.get_env_resources(), config=self.config, is_add_action=False,
            action_space_wrapper=self.action_space_wrapper
        )

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        # self._choose_full_action(resource, nodes, demand)
        # if self.current_add_node is not None:
        #     return self.current_add_node
        if self.use_pool_node:
            self.current_add_node = local_optimal_add_node_policy_with_pool(
                self.action_space_wrapper.add_node_space, demand, resource, nodes,
                self.config.environment.get_env_resources(),
                self.config.environment.reward.parameters['delta_target'],
                self.config.environment.reward.parameters['gap_with_units']
            )
        else:
            self.current_add_node = local_optimal_add_node_policy(
                state, self.action_space_wrapper.add_node_space, demand, resource, nodes, self.add_node_with_node_groups
            )
        return self.current_add_node

    def _choose_node_remove(self, state: State, epsilon: Optional[float] = None,
                            random: Optional[float] = None, resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:
        if self.config.environment.sub_agents_setup.full_action_split:
            combined_action = self._choose_movement_arguments(state, self.current_add_node, resource, nodes, demand)
            remove_node, _, _ = self._handle_combined_action(combined_action)
            return remove_node
        return self.current_remove_node

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        if self.current_combined is not None:
            return self._handle_combined_action(self.current_combined)
        combined_action = self._choose_movement_arguments(state, add_node, resource, nodes, demand)
        return self._handle_combined_action(combined_action)

    # def _choose_resource_class(self, state: State, epsilon: Optional[float] = None,
    #                            random: Optional[float] = None, resource: Optional[str] = None) -> int:
    #     return self.current_resource_class

    def _choose_quantity(self, state: State, epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        return self.current_quantity

    def _choose_movement_arguments(self, state: State, add_node_index: Optional[int], resource: str, nodes: List[Node],
                                   demand: StepData) -> int:
        if self.use_pool_node:
            return movement_local_optimal_policy_with_pool(
                add_node_index=add_node_index, resource=resource, nodes=nodes, demand=demand,
                action_space_wrapper=self.action_space_wrapper,
                quantity_movements=self.quantity_movements,
                resource_classes=self.resource_classes,
                config=self.config,
                resources=self.config.environment.get_env_resources(),
                delta_target=self.config.environment.reward.parameters['delta_target']
            )
        else:
            return new_movement_local_optimal_policy(
                add_node_index=add_node_index, resource=resource, nodes=nodes, demand=demand,
                action_space_wrapper=self.action_space_wrapper,
                quantity_movements=self.quantity_movements,
                resource_classes=self.resource_classes,
                config=self.config,
                remaining_gap_max_factor=self.remaining_gap_max_factor[resource],
                max_cost=self.max_cost[resource]
            )
        # return local_optimal_movement_policy(
        #     state, add_node_index, resource, nodes, demand,
        #     movement_action_space=self.action_space_wrapper.combined_space,
        #     add_node_action_space=self.action_space_wrapper.add_node_space,
        #     quantity_movements=self.quantity_movements,
        #     resource_classes=self.resource_classes,
        #     is_conservative=self.is_conservative
        # )

    def _choose_full_action(self, resource: str, nodes: List[Node], demand: StepData):
        action: Action = new_full_action_greedy_policy(
            resource, nodes, demand,
            action_space_wrapper=self.action_space_wrapper, quantity_movements=self.quantity_movements,
            resource_classes=self.resource_classes,
            config=self.config,
            remaining_gap_max_factor=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource]
        )
        self.current_add_node = action.add_node
        self.current_combined = self.action_space_wrapper.combined_inverted_mapping[(
            action.remove_node, action.resource_class, action.quantity
        )]


class SamplingOptimalAgent(BaselineAgent):

    def __init__(
            self,
            action_space_wrapper: ActionSpaceWrapper,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            **kwargs
    ):
        super(SamplingOptimalAgent, self).__init__(action_space_wrapper,
                                                   random_state=random_state,
                                                   name=AgentType.SamplingOptimal,
                                                   action_spaces=action_spaces,
                                                   state_spaces=state_spaces,
                                                   log=log,
                                                   config=config,
                                                   mode=mode,
                                                   **kwargs)
        self.batch_size: int = self.config.environment.agent.sampling_optimal.batch_size
        self.n_nodes = self.config.environment.nodes.get_n_nodes()
        self.n_resource_classes = len(self.config.environment.resources[0].classes)
        self.resource_classes: Dict[str, ResourceClass] = self.config.environment.resources[0].classes
        self.quantity_movements: List[int] = self.config.environment.action_space.bucket_move
        self.use_pool_node: bool = self.config.environment.nodes.use_pool_node

        self.current_add_node_action_index: Optional[int] = None
        self.current_combined_action_index: Optional[int] = None
        self.current_add_action: Optional[int] = None
        self.current_remove_action: Optional[int] = None

    def _get_possible_actions(self) -> List[Tuple[int, int]]:
        add_node_space = self.action_space_wrapper.add_node_space.get_available_actions()
        if self.action_space_wrapper.add_node_space.is_wait_action(add_node_space[-1]):
            add_node_space = add_node_space[: -1]
        movement_space = self.action_space_wrapper.combined_space.get_available_actions()
        if self.action_space_wrapper.combined_space.is_wait_action(movement_space[-1]):
            movement_space = movement_space[: -1]
        actions = list(product(add_node_space, movement_space))
        actions.append((
            self.action_space_wrapper.add_node_space.wait_action_index,
            self.action_space_wrapper.combined_space.wait_action_index
        ))
        return actions

    def _get_pool_node_index(self, nodes: List[Node]) -> Optional[int]:
        for i, node in enumerate(nodes):
            if node.base_station_type == POOL_NODE:
                return i
        return None

    def _get_nodes_deltas(
            self,
            nodes: List[Node],
            demand: StepData,
            resource: str,
            gap_with_units: bool,
            resources: List[EnvResource],
    ) -> List[float]:
        deltas: List[float] = []
        res_capacity = 1
        if gap_with_units:
            for env_res in resources:
                if env_res.name == resource:
                    for _, res_info in env_res.classes.items():
                        res_capacity = res_info['capacity']
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        for i, node in enumerate(nodes):
            if node.base_station_type != POOL_NODE:
                node_demand = resource_demand[i][node.base_station_type]
                current_allocated = node.get_current_allocated(resource)
                delta = current_allocated - node_demand
                delta = delta / res_capacity
                delta = math.floor(delta)
                deltas.append(delta)
        return deltas

    def _action_is_valid(
            self,
            add_node_action_index: int,
            combined_action_index: int,
            nodes: List[Node],
            node_groups: NodeGroups,
            resource: str
    ) -> bool:
        add_node_index = -1
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node_action_index):
            if self.add_node_with_node_groups:
                add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node_action_index]
                add_node_index = node_groups.sample_group(group_name=add_node_group, resource_name=resource)
            else:
                add_node_index = add_node_action_index
        remove_node_index, resource_class, quantity = self.action_space_wrapper.combined_mapping[combined_action_index]
        resource_class_name = self.action_space_wrapper.resource_classes_space.actions_mapping[resource_class]
        proposed_actions = Action(add_node_action_index, remove_node_index, resource_class, quantity)
        quantity_val = self.action_space_wrapper.quantity_space.actions_mapping[quantity]
        if self.action_space_wrapper.is_wait_action(proposed_actions):
            return True
        if add_node_index == remove_node_index:
            return False
        pool_node_index = self._get_pool_node_index(nodes)
        if self.use_pool_node and add_node_index != pool_node_index:
            if remove_node_index != pool_node_index:
                return False
            else:
                pool_node = nodes[pool_node_index]
                pool_units = pool_node.get_current_resource_class_units(resource, res_class=resource_class_name)
                if pool_units >= quantity_val:
                    return True
                else:
                    return False
        else:
            remove_node = nodes[remove_node_index]
            node_units = remove_node.get_current_resource_class_units(resource, res_class=resource_class_name)
            if node_units >= quantity_val:
                return True
            else:
                return False

    def _compute_action_cost(self, nodes: List[Node], resource: str, action: Action) -> float:
        resource_classes = nodes[0].initial_resources[resource].classes
        action_resource_class = self.action_space_wrapper.resource_classes_space.actions_mapping[action.resource_class]
        if self.action_space_wrapper.is_wait_action(action):
            units_moved = 0
            resource_unit_cost = 0
        else:
            units_moved = self.action_space_wrapper.quantity_space.actions_mapping[action.quantity]
            resource_unit_cost = resource_classes[action_resource_class]['cost']
        return units_moved * resource_unit_cost

    def _estimate_utility(
            self,
            add_node_action_index: int,
            combined_action_index: int,
            resource: str,
            nodes: List[Node],
            demand: StepData,
            node_groups: NodeGroups,
    ) -> float:
        # if action is wait, compute directly the remaining gap only and return it scaled based on alpha
        # else compute the cost of the action and the remaining gap, finally scale them using alpha
        alpha: float = self.config.environment.reward.parameters['alpha']
        weights: List[float] = self.config.environment.reward.parameters['weights']
        scalarization_function: ScalarizationFunction = self.config.environment.reward.parameters[
            'scalarization_function']
        normalization_range: Tuple[int, int] = self.config.environment.reward.parameters['val_eval_normalization_range']
        remove_node_index, resource_class, quantity = self.action_space_wrapper.combined_mapping[combined_action_index]
        gap_with_units = self.config.environment.reward.parameters['gap_with_units']
        units = 1
        if gap_with_units:
            for res_name, res_info in self.resource_classes.items():
                units = res_info['capacity']
        add_node_index = -1
        if not self.action_space_wrapper.add_node_space.is_wait_action(add_node_action_index):
            if self.add_node_with_node_groups:
                add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node_action_index]
                add_node_index = node_groups.sample_group(group_name=add_node_group, resource_name=resource)
            else:
                add_node_index = add_node_action_index
        proposed_action = Action(add_node_action_index, remove_node_index, resource_class, quantity)
        is_wait = self.action_space_wrapper.is_wait_action(proposed_action)
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        resource_classes = nodes[0].initial_resources[resource].classes
        action_resource_class = self.action_space_wrapper.resource_classes_space.actions_mapping[resource_class]
        capacity_moved = 0
        if not is_wait:
            units_moved = self.action_space_wrapper.quantity_space.actions_mapping[quantity]
            capacity_moved = units_moved * resource_classes[action_resource_class]['capacity'] / units
        # compute the remaining gap
        gap = 0
        pre_gap = 0
        for i, node in enumerate(nodes):
            node_demand = np.ceil(resource_demand[i][node.base_station_type] / units)
            current_allocated = node.get_current_allocated(resource) / units
            pre_diff = math.floor(current_allocated - node_demand)
            if pre_diff < 0:
                pre_gap += pre_diff
            if not is_wait:
                if i == add_node_index:
                    # add capacity
                    current_allocated += capacity_moved
                if i == remove_node_index:
                    # remove capacity
                    current_allocated -= capacity_moved
            difference = current_allocated - node_demand
            difference = math.floor(difference)
            if difference < 0:
                gap += difference
        action_cost = self._compute_action_cost(nodes, resource, proposed_action)
        values = [gap, action_cost, pre_gap]
        if self.config.environment.reward.parameters['normalize_objectives']:
            normalized_post_gap = normalize_scalar(gap, 0,
                                                   self.remaining_gap_max_factor[resource],
                                                   normalization_range[0], normalization_range[1])
            normalized_cost = normalize_scalar(action_cost, 0,
                                               self.max_cost[resource],
                                               normalization_range[0], normalization_range[1])
            normalized_pre_gap = normalize_scalar(pre_gap, 0,
                                                  self.remaining_gap_max_factor[resource],
                                                  normalization_range[0], normalization_range[1])
            values = [normalized_post_gap, normalized_cost, normalized_pre_gap]
        utility = scalarize_rewards(
            func_type=scalarization_function,
            remaining_gap=gap,
            cost=action_cost,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            values=values,
            alpha=alpha,
            weights=weights,
            normalization_range=normalization_range,
            normalize_objectives=self.config.environment.reward.parameters['normalize_objectives']
        )
        return utility

    def _resource_utilization_estimate(
            self,
            add_node_action_index: int,
            combined_action_index: int,
            resource: str,
            nodes: List[Node],
            demand: StepData,
    ) -> float:
        # config parameters
        pool_node_index = self._get_pool_node_index(nodes)
        delta_target = self.config.environment.reward.parameters['delta_target']
        alpha: float = self.config.environment.reward.parameters['alpha']
        scalarization_function: ScalarizationFunction = self.config.environment.reward.parameters[
            'scalarization_function']
        normalization_range: Tuple[int, int] = self.config.environment.reward.parameters['val_eval_normalization_range']
        success_reward = self.config.environment.reward.parameters['val_eval_success_reward']
        # initial deltas
        deltas = self._get_nodes_deltas(nodes, demand, resource,
                                        self.config.environment.reward.parameters['gap_with_units'],
                                        self.config.environment.get_env_resources())
        deltas.insert(pool_node_index, delta_target)
        deltas = np.array(deltas)
        # get the sub-parts of the movement action
        remove_node_index, resource_class, quantity = self.action_space_wrapper.combined_mapping[combined_action_index]
        proposed_action = Action(add_node_action_index, remove_node_index, resource_class, quantity)
        # get the proposed action cost
        action_cost = self._compute_action_cost(nodes, resource, proposed_action)
        is_wait = self.action_space_wrapper.is_wait_action(proposed_action)
        units_moved = 0
        new_deltas = np.copy(deltas)
        if not is_wait:
            units_moved = self.action_space_wrapper.quantity_space.actions_mapping[quantity]
            # update deltas
            if add_node_action_index != pool_node_index:
                new_deltas[add_node_action_index] += units_moved
            if remove_node_index != pool_node_index:
                new_deltas[remove_node_index] -= units_moved
        distance = l1_distance(new_deltas, delta_target).sum()
        if distance == delta_target:
            distances_normalized = success_reward
        else:
            distances_normalized = normalize_scalar(-distance, 0, self.remaining_gap_max_factor[resource] * 2,
                                                    a=normalization_range[0], b=normalization_range[1])
        action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                  a=normalization_range[0], b=normalization_range[1])
        utility = scalarize_rewards(
            func_type=scalarization_function,
            remaining_gap=distances_normalized,
            cost=action_cost_normalized,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            alpha=alpha,
            normalize_objectives=False,
            only_apply_scalarization=True
        )
        return utility

    def _gap_from_deltas(self, deltas: np.array, delta_target: float) -> float:
        gap = 0
        for val in deltas:
            if val < delta_target:
                gap += (delta_target - val)
        if gap > 0:
            gap = gap * -1
        return gap

    def _surplus_from_deltas(self, deltas: np.array, delta_target: float) -> float:
        exceeding = 0
        for val in deltas:
            if val > delta_target:
                exceeding += (val - delta_target)
        return exceeding

    def _gap_surplus_cost_estimate(
            self,
            add_node_action_index: int,
            combined_action_index: int,
            resource: str,
            nodes: List[Node],
            demand: StepData,
    ) -> float:
        # config parameters
        pool_node_index = self._get_pool_node_index(nodes)
        delta_target = self.config.environment.reward.parameters['delta_target']
        weights: float = self.config.environment.reward.parameters['weights']
        normalization_range: Tuple[int, int] = self.config.environment.reward.parameters['val_eval_normalization_range']
        # initial deltas
        deltas = self._get_nodes_deltas(nodes, demand, resource,
                                        self.config.environment.reward.parameters['gap_with_units'],
                                        self.config.environment.get_env_resources())
        deltas.insert(pool_node_index, 0)
        deltas = np.array(deltas)
        # get the sub-parts of the movement action
        remove_node_index, resource_class, quantity = self.action_space_wrapper.combined_mapping[combined_action_index]
        proposed_action = Action(add_node_action_index, remove_node_index, resource_class, quantity)
        # get the proposed action cost
        action_cost = self._compute_action_cost(nodes, resource, proposed_action)
        is_wait = self.action_space_wrapper.is_wait_action(proposed_action)
        units_moved = 0
        new_deltas = np.copy(deltas)
        if not is_wait:
            units_moved = self.action_space_wrapper.quantity_space.actions_mapping[quantity]
            # update deltas
            if add_node_action_index != pool_node_index:
                new_deltas[add_node_action_index] += units_moved
            if remove_node_index != pool_node_index:
                new_deltas[remove_node_index] -= units_moved
        remaining_gap = self._gap_from_deltas(new_deltas, delta_target)
        surplus = self._surplus_from_deltas(new_deltas, delta_target)
        gap_normalized = normalize_scalar(remaining_gap, 0, self.remaining_gap_max_factor[resource],
                                          a=normalization_range[0], b=normalization_range[1])
        surplus_normalized = normalize_scalar(-surplus, 0, self.remaining_gap_max_factor[resource],
                                              a=normalization_range[0], b=normalization_range[1])
        action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                  a=normalization_range[0], b=normalization_range[1])
        values = [gap_normalized, surplus_normalized, action_cost_normalized]
        utility = scalarize_rewards(
            func_type=ScalarizationFunction.LinearWeights,
            remaining_gap=gap_normalized,
            cost=action_cost_normalized,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            values=values,
            weights=weights,
        )
        return utility

    def _sample_available_actions(
            self, available_actions: List[Tuple[int, int]], batch_size: int,
            nodes: List[Node], node_groups: NodeGroups, resource: str,
            max_tries: int = 5,
            current_try: int = 0
    ):
        if current_try > max_tries:
            return []
        sampled = available_actions
        remaining_actions = available_actions
        original_length = len(available_actions)
        if len(sampled) >= self.batch_size:
            sampled_indexes = self.random.choice(list(range(len(available_actions))), batch_size, replace=False)
            sampled = []
            remaining_actions = []
            for i, action in enumerate(available_actions):
                if i in sampled_indexes:
                    sampled.append(action)
                else:
                    remaining_actions.append(action)
            original_length = batch_size
        if original_length > len(available_actions):
            original_length = len(available_actions)
        result_sampled = []
        for a in sampled:
            is_valid = self._action_is_valid(a[0], a[1], nodes, node_groups, resource)
            if is_valid:
                result_sampled.append(a)
        if len(result_sampled) < original_length:
            result_sampled += self._sample_available_actions(remaining_actions,
                                                             original_length - len(result_sampled),
                                                             nodes, node_groups, resource, current_try=current_try + 1
                                                             )
        unique = []
        for a in result_sampled:
            if a not in unique:
                unique.append(a)
        return unique

    def _choose_action(self, resource: str, node_groups: NodeGroups, nodes: List[Node], demand: StepData):
        self.action_space_wrapper.unmask_all()
        available_actions = self._get_possible_actions()
        sampled = self._sample_available_actions(available_actions, self.batch_size, nodes, node_groups, resource)

        max_utility: Optional[float] = None
        best_actions: Optional[Tuple[int, int]] = None
        for s_action in sampled:
            # check the utility and choose the one with the highest
            add_node_action_index = s_action[0]
            combined = s_action[1]
            if self.use_pool_node and self.config.environment.reward.type == RewardFunctionType.ResourceUtilization:
                utility = self._resource_utilization_estimate(add_node_action_index, combined, resource, nodes, demand)
            elif self.use_pool_node and self.config.environment.reward.type == RewardFunctionType.GapSurplusCost:
                utility = self._gap_surplus_cost_estimate(add_node_action_index, combined, resource, nodes, demand)
            else:
                utility = self._estimate_utility(add_node_action_index, combined, resource, nodes, demand, node_groups)
            if max_utility is None or utility >= max_utility:
                max_utility = utility
                best_actions = (add_node_action_index, combined)
        if best_actions is not None:
            self.current_add_node_action_index = best_actions[0]
            self.current_combined_action_index = best_actions[1]
        else:
            self.current_add_node_action_index = self.action_space_wrapper.add_node_space.wait_action_index
            self.current_combined_action_index = self.action_space_wrapper.combined_space.wait_action_index

    def _choose_node_remove(self, state: State, epsilon: Optional[float] = None,
                            random: Optional[float] = None, resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:

        if self.config.environment.sub_agents_setup.full_action_split:
            remove_node, _, _ = self._handle_combined_action(self.current_combined_action_index)
            return remove_node
        return self.current_remove_node

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        self._choose_action(resource, node_groups=node_groups, nodes=nodes, demand=demand)
        return self.current_add_node_action_index

    def _choose_combined_sub_action(self, state: State, add_node: int, nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None, demand: Optional[StepData] = None
                                    ) -> Tuple[int, int, int]:
        return self._handle_combined_action(self.current_combined_action_index)

    # def _choose_resource_class(self, state: State, epsilon: Optional[float] = None,
    #                            random: Optional[float] = None, resource: Optional[str] = None) -> int:
    #     return self.current_resource_class

    def _choose_quantity(self, state: State, epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        return self.current_quantity

    def _sample_action_space(self, action_space: ActionSpace, batch_size: int) -> np.ndarray:
        available_actions = action_space.get_available_actions()
        if batch_size == -1 or len(available_actions) < batch_size:
            return available_actions
        else:
            return self.random.choice(available_actions, batch_size, replace=False)

    def _estimate_resource_utilization_with_pool(
            self,
            add_action: int,
            remove_action: int,
            nodes: List[Node],
            resource: str,
            demand: StepData,
    ) -> float:
        delta_target = self.config.environment.reward.parameters['delta_target']
        alpha: float = self.config.environment.reward.parameters['alpha']
        scalarization_function: ScalarizationFunction = self.config.environment.reward.parameters[
            'scalarization_function']
        normalization_range: Tuple[int, int] = self.config.environment.reward.parameters['val_eval_normalization_range']
        action_cost = 0
        deltas = self._get_nodes_deltas(nodes, demand, resource,
                                        self.config.environment.reward.parameters['gap_with_units'],
                                        self.config.environment.get_env_resources())
        deltas = np.array(deltas)
        if not self.action_space_wrapper.add_action_space.is_wait_action(add_action):
            deltas[add_action] += 1
            action_cost += 1
        if not self.action_space_wrapper.remove_action_space.is_wait_action(remove_action):
            deltas[remove_action] -= 1
            action_cost += 1

        distance = l1_distance(deltas, delta_target).sum()
        distances_normalized = normalize_scalar(-distance, 0, self.remaining_gap_max_factor[resource] * 2,
                                                a=normalization_range[0], b=normalization_range[1])
        action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                  a=normalization_range[0], b=normalization_range[1])
        utility = scalarize_rewards(
            func_type=scalarization_function,
            remaining_gap=distances_normalized,
            cost=action_cost_normalized,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            alpha=alpha,
            normalize_objectives=False,
            only_apply_scalarization=True
        )
        return utility

    def _estimate_gap_surplus_cost_with_pool(
            self,
            add_action: int,
            remove_action: int,
            nodes: List[Node],
            resource: str,
            demand: StepData,
    ) -> float:
        weights: float = self.config.environment.reward.parameters['weights']
        delta_target = self.config.environment.reward.parameters['delta_target']
        normalization_range: Tuple[int, int] = self.config.environment.reward.parameters['val_eval_normalization_range']
        action_cost = 0
        deltas = self._get_nodes_deltas(nodes, demand, resource,
                                        self.config.environment.reward.parameters['gap_with_units'],
                                        self.config.environment.get_env_resources())
        deltas = np.array(deltas)
        # get the sub-parts of the movement action
        if not self.action_space_wrapper.add_action_space.is_wait_action(add_action):
            deltas[add_action] += 1
            action_cost += 1
        if not self.action_space_wrapper.remove_action_space.is_wait_action(remove_action):
            deltas[remove_action] -= 1
            action_cost += 1
        remaining_gap = self._gap_from_deltas(deltas, delta_target)
        surplus = self._surplus_from_deltas(deltas, delta_target)
        gap_normalized = normalize_scalar(remaining_gap, 0, self.remaining_gap_max_factor[resource],
                                          a=normalization_range[0], b=normalization_range[1])
        surplus_normalized = normalize_scalar(-surplus, 0, self.remaining_gap_max_factor[resource],
                                              a=normalization_range[0], b=normalization_range[1])
        action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                  a=normalization_range[0], b=normalization_range[1])
        values = [gap_normalized, surplus_normalized, action_cost_normalized]
        utility = scalarize_rewards(
            func_type=ScalarizationFunction.LinearWeights,
            remaining_gap=gap_normalized,
            cost=action_cost_normalized,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            values=values,
            weights=weights,
        )
        return utility

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        self.action_space_wrapper.unmask_all()
        add_actions = self._sample_action_space(self.action_space_wrapper.add_action_space, self.batch_size)
        remove_actions = self._sample_action_space(self.action_space_wrapper.remove_action_space, self.batch_size)
        actions = list(product(add_actions, remove_actions))

        max_utility: Optional[float] = None
        best_actions: List[Tuple[int, int]] = []
        for (add_action, remove_action) in actions:
            if pool_node.get_current_allocated(resource) > 0 or self.action_space_wrapper.add_action_space.is_wait_action(add_action):
                if self.config.environment.reward.type == RewardFunctionType.GapSurplusCost:
                    utility = self._estimate_gap_surplus_cost_with_pool(add_action, remove_action, nodes, resource, demand)
                else:
                    utility = self._estimate_resource_utilization_with_pool(add_action, remove_action, nodes, resource, demand)
                if max_utility is None or utility >= max_utility:
                    if utility == max_utility:
                        best_actions.append((add_action, remove_action))
                    else:
                        max_utility = utility
                        best_actions = [(add_action, remove_action)]

        if len(best_actions) > 0:
            index = self.random.randint(0, len(best_actions))
            self.current_add_action = best_actions[index][0]
            self.current_remove_action = best_actions[index][1]
        else:
            self.current_add_action = self.action_space_wrapper.add_node_space.wait_action_index
            self.current_remove_action = self.action_space_wrapper.remove_node_space.wait_action_index
        return self.current_add_action

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        return self.current_remove_action


class ExhaustiveSearchAgent(SamplingOptimalAgent):

    def __init__(
            self,
            action_space_wrapper: ActionSpaceWrapper,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            **kwargs
    ):
        super(ExhaustiveSearchAgent, self).__init__(action_space_wrapper,
                                                    random_state=random_state,
                                                    action_spaces=action_spaces,
                                                    state_spaces=state_spaces,
                                                    log=log,
                                                    config=config,
                                                    mode=mode,
                                                    **kwargs)
        self.name = AgentType.Greedy
        self.batch_size = -1

    def _choose_action(self, resource: str, node_groups: NodeGroups, nodes: List[Node], demand: StepData):
        self.action_space_wrapper.unmask_all()
        possible_actions = self._get_possible_actions()
        available_actions = []
        for action in possible_actions:
            if self._action_is_valid(action[0], action[1], nodes, node_groups, resource):
                available_actions.append(action)

        max_utility: Optional[float] = None
        best_actions: Optional[Tuple[int, int]] = None
        for s_action in available_actions:
            # check the utility and choose the one with the highest
            add_node_action_index = s_action[0]
            combined = s_action[1]
            if self.use_pool_node and self.config.environment.reward.type == RewardFunctionType.ResourceUtilization:
                utility = self._resource_utilization_estimate(add_node_action_index, combined, resource, nodes, demand)
            elif self.use_pool_node and self.config.environment.reward.type == RewardFunctionType.GapSurplusCost:
                utility = self._gap_surplus_cost_estimate(add_node_action_index, combined, resource, nodes, demand)
            else:
                utility = self._estimate_utility(add_node_action_index, combined, resource, nodes, demand, node_groups)
            if max_utility is None or utility > max_utility:
                max_utility = utility
                best_actions = (add_node_action_index, combined)
        self.current_add_node_action_index = best_actions[0]
        self.current_combined_action_index = best_actions[1]
