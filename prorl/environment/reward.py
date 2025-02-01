import math
from abc import abstractmethod
from typing import List, Dict, Union, Any, Optional, Tuple

import numpy as np

from prorl.common.enum_utils import ExtendedEnum
from prorl.common.math_util import normalize_scalar, l1_distance
from prorl.core.state import State
from prorl.core.step import Step
from prorl.core.step_data import StepData
from prorl.common.data_structure import RunMode
from prorl.environment.action_space import Action, ActionSpaceWrapper
from prorl.environment.data_structure import EnvResource
from prorl.environment.node import Node, POOL_NODE
from prorl.environment.node_groups import NodeGroups
from prorl.environment.scalarization_function import ScalarizationFunction, scalarize_rewards


class RewardFunctionType(str, ExtendedEnum):
    GlobalRemainingGap = 'global-remaining-gap'
    GlobalSatisfiedNodes = 'global-satisfied-nodes'
    GlobalThreeObjectives = 'global-3-objectives'
    MultiObjectivesSimple = 'multi-objectives-simple'
    ResourceUtilization = 'resource-utilization'
    GapSurplusCost = 'gap-surplus-cost'


class RewardFunctionTypeError(Exception):

    def __init__(self, name: RewardFunctionType, *args):
        message = f'No reward class mapping for type: {name}'
        super(RewardFunctionTypeError, self).__init__(message, *args)


class RewardAbstract:

    def __init__(self, resources: List[EnvResource], name: str, env_config,
                 run_mode: RunMode, disable_cost: bool = False, gap_with_units: bool = False,
                 action_per_step: int = 1,
                 **kwargs):
        self.name: str = name
        self.env_config = env_config
        self.resources: List[EnvResource] = resources
        self.gap_with_units: bool = gap_with_units
        self.disable_cost: bool = disable_cost
        self.resources_map: Dict[str, EnvResource] = {r.name: r for r in self.resources}
        self.resources_total: Dict[str, int] = {}
        self.null_reward: float = self.env_config.reward.null_reward
        self.null_between_interval: Optional[int] = self.env_config.reward.null_between_interval
        self.run_mode: RunMode = run_mode
        self.actions_per_step = action_per_step
        for resource in self.resources:
            self.resources_total[resource.name] = resource.total_available
        self.max_cost: Dict[str, float] = {}
        max_quantity = max(self.env_config.action_space.bucket_move)
        for res_info in self.resources:
            max_cost_res = 0
            for c_name, c_info in res_info.classes.items():
                if c_info['cost'] * max_quantity > max_cost_res:
                    max_cost_res = c_info['cost'] * max_quantity
            self.max_cost[res_info.name] = max_cost_res * self.actions_per_step
        self.remaining_gap_max_factor: Dict[str, float] = {}
        self.step_action_cost = 0

    def __str__(self):
        return f'<Reward name={self.name} >'

    def is_a_null_step(self, step: Step) -> bool:
        if step.total_steps % self.env_config.reward.null_between_interval != 0:
            return True
        else:
            return False

    def set_remaining_gap_max_factor(self, resource: str, value: float):
        if self.gap_with_units:
            res_units = 0
            n_nodes = self.env_config.nodes.get_n_nodes(no_pool=True)
            for res in self.resources:
                if res.name == resource:
                    # for _, info in res.classes.items():
                    #     res_units += info['allocated']  # * n_nodes
                    res_units = res.total_units / n_nodes
                self.remaining_gap_max_factor[resource] = -res_units
        else:
            self.remaining_gap_max_factor[resource] = value

    def reward_info(self) -> Optional[Dict[str, Any]]:
        return None

    def update_reward_info_for_null_reward(self):
        pass

    def set_initial_gap(self, nodes: List[Node], demand: StepData, resource: str):
        if resource not in self.remaining_gap_max_factor:
            self.remaining_gap_max_factor[resource] = self.compute_remaining_gap(nodes, demand, resource)

    def compute(
            self,
            state_wrapper: Dict[str, State],
            actions: List[Action],
            step: Step,
            demand: StepData,
            nodes: List[Node],
            resource: str,
            penalty: Union[int, float] = 0,
            current_budget=0,
            action_space_wrapper: Optional[ActionSpaceWrapper] = None,
            node_groups: Optional[NodeGroups] = None,
            **kwargs
    ) -> Union[float, Tuple[float, float, float]]:
        reward = self._compute(state_wrapper, actions, step, demand, nodes, resource, penalty, current_budget,
                               action_space_wrapper, node_groups, **kwargs)
        if self.null_between_interval is not None and step.total_steps % self.null_between_interval != 0:
            reward = self.null_reward
            self.update_reward_info_for_null_reward()
        if penalty > 0:
            reward = penalty
        if self.env_config.sub_agents_setup.same_reward:
            if isinstance(reward, tuple):
                return reward[0]
            else:
                return reward
        else:
            return reward

    @abstractmethod
    def _compute(
            self,
            state_wrapper: Dict[str, State],
            actions: List[Action],
            step: Step,
            demand: StepData,
            nodes: List[Node],
            resource: str,
            penalty: Union[int, float] = 0,
            current_budget=0,
            action_space_wrapper: Optional[ActionSpaceWrapper] = None,
            node_groups: Optional[NodeGroups] = None,
            **kwargs
    ) -> Union[float, Tuple[float, float]]:
        pass

    def reset(self):
        pass

    def compute_action_cost(self, nodes: List[Node], resource: str,
                            actions: List[Action],
                            action_space_wrapper: Optional[ActionSpaceWrapper]) -> float:
        cost = 0
        for action in actions:
            resource_classes = nodes[0].initial_resources[resource].classes
            action_resource_class = action_space_wrapper.resource_classes_space.actions_mapping[action.resource_class]
            if action_space_wrapper.is_wait_action(action):
                units_moved = 0
                resource_unit_cost = 0
            else:
                units_moved = action_space_wrapper.quantity_space.actions_mapping[action.quantity]
                resource_unit_cost = resource_classes[action_resource_class]['cost']
            cost += (units_moved * resource_unit_cost)
        return cost

    def compute_node_delta_reward(self, demand: StepData, resource: str, action: Action,
                                  action_space_wrapper: ActionSpaceWrapper, nodes: List[Node], node_index) -> float:
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        if not action_space_wrapper.is_wait_action(action):
            node: Node = nodes[node_index]
            node_demand: float = resource_demand[node_index][node.base_station_type]
            difference = math.floor(node.get_current_allocated(resource) - node_demand)
            if difference < 0:
                return difference
            else:
                return 0
        else:
            return 0

    def _remove_node_delta(self, action: Action, demand: StepData, resource: str,
                           node_allocated: Union[int, float],
                           nodes: List[Node],
                           action_space_wrapper: ActionSpaceWrapper) -> float:
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        remove_node_index = action_space_wrapper.remove_node_space.actions_mapping[action.remove_node]
        if not action_space_wrapper.is_wait_action(action):
            node: Node = nodes[remove_node_index]
            node_demand: float = resource_demand[remove_node_index][node.base_station_type]
            difference = math.floor(node_allocated - node_demand)
            if difference < 0:
                return difference
            else:
                return 0
        else:
            return 0

    def get_satisfied_nodes(self, demand: StepData, nodes: List[Node], resource: str) -> int:
        value = 0
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        for i in range(len(nodes)):
            node: Node = nodes[i]
            node_demand: float = resource_demand[i][node.base_station_type]
            node_current_allocation = node.get_current_allocated(resource)
            difference = node_current_allocation - node_demand
            if difference >= 0:
                value += 1
        return value

    def compute_remaining_gap(
            self,
            nodes: List[Node],
            demand: StepData,
            resource: str
    ) -> float:
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        gap = 0.0
        for i, node in enumerate(nodes):
            node_demand = resource_demand[i][node.base_station_type]
            current_allocated = node.get_current_allocated(resource)
            difference = current_allocated - node_demand
            difference = math.floor(difference)
            if difference < 0:
                gap += difference
        if self.gap_with_units:
            res_capacity = 1
            for env_res in self.resources:
                if env_res.name == resource:
                    for _, res_info in env_res.classes.items():
                        res_capacity = res_info['capacity']
            gap = np.floor(gap / res_capacity)
        return gap


class GlobalRemainingGapReward(RewardAbstract):

    def __init__(
            self,
            resources: List[EnvResource],
            env_config,
            run_mode: RunMode,
            disable_cost: bool = False,
            alpha=0.9,
            scalarized: bool = True,
            clipped_remaining_gap: bool = False,
            normalize_objectives: bool = True,
            training_normalization_range: Tuple[int, int] = (0, 1),
            val_eval_normalization_range: Tuple[int, int] = (0, 1),
            scalarization_function: ScalarizationFunction = ScalarizationFunction.Linear,
            zero_remaining_gap_reward: float = 0,
            action_per_step: int = 1,
            **kwargs
    ):
        super(GlobalRemainingGapReward, self).__init__(resources=resources, env_config=env_config,
                                                       disable_cost=disable_cost,
                                                       run_mode=run_mode,
                                                       name=RewardFunctionType.GlobalRemainingGap,
                                                       action_per_step=action_per_step, **kwargs)
        self.alpha = alpha
        self.scalarized = scalarized
        self.clipped_remaining_gap = clipped_remaining_gap
        if self.run_mode != RunMode.Train:
            self.clipped_remaining_gap = False
        self.last_remaining_gap: Optional[float] = None
        self.last_cost: Optional[float] = None
        self.training_normalization_range: Tuple[int, int] = training_normalization_range
        self.val_eval_normalization_range: Tuple[int, int] = val_eval_normalization_range
        self.normalize_objectives: bool = normalize_objectives
        self.scalarization_function: ScalarizationFunction = scalarization_function
        self.zero_remaining_gap_reward: float = zero_remaining_gap_reward

    @property
    def normalization_range(self) -> Tuple[int, int]:
        if self.run_mode == RunMode.Train:
            return self.training_normalization_range
        else:
            return self.val_eval_normalization_range

    def reward_info(self) -> Optional[Dict[str, Any]]:
        if self.last_remaining_gap is not None and self.last_cost is not None:
            return {
                'remaining_gap': self.last_remaining_gap,
                'cost': self.last_cost
            }

    def _compute(self, state_wrapper: Dict[str, State], actions: List[Action], step: Step, demand: StepData, nodes: List[Node],
                 resource: str, penalty: Union[int, float] = 0, current_budget=0,
                 action_space_wrapper: Optional[ActionSpaceWrapper] = None, node_groups: Optional[NodeGroups] = None,
                 **kwargs) -> Union[float, Tuple[float, float]]:
        action_cost = self.compute_action_cost(nodes, resource, actions, action_space_wrapper)
        if self.disable_cost:
            action_cost = 0
        remaining_gap = self.compute_remaining_gap(nodes, demand, resource)
        self.last_cost = action_cost
        self.last_remaining_gap = remaining_gap
        if remaining_gap == 0:
            remaining_gap = self.zero_remaining_gap_reward
        reward = scalarize_rewards(
            func_type=self.scalarization_function,
            remaining_gap=remaining_gap,
            cost=action_cost,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            alpha=self.alpha,
            clipped_remaining_gap=self.clipped_remaining_gap,
            normalization_range=self.normalization_range,
            normalize_objectives=self.normalize_objectives,
            satisfied_nodes=self.get_satisfied_nodes(demand, nodes, resource),
            n_nodes=len(nodes)
        )
        return reward


# class MultiObjectivesSimpleReward(RewardAbstract):
#
#     def __init__(
#             self,
#             resources: List[EnvResource],
#             env_config,
#             run_mode: RunMode,
#             disable_cost: bool = False,
#             alpha=0.9,
#             scalarization_function: ScalarizationFunction = ScalarizationFunction.Linear,
#             hour_satisfied_bonus: float = 0,
#             satisfied_or_nothing: bool = False,
#             **kwargs
#     ):
#         super(MultiObjectivesSimpleReward, self).__init__(resources=resources, env_config=env_config,
#                                                           disable_cost=disable_cost,
#                                                           run_mode=run_mode,
#                                                           name=RewardFunctionType.MultiObjectivesSimple, **kwargs)
#         self.alpha: float = alpha
#         self.scalarization_function: ScalarizationFunction = scalarization_function
#         self.last_pre_gap: Optional[float] = None
#         self.last_remaining_gap: Optional[float] = None
#         self.last_cost: Optional[float] = None
#         self.hour_satisfied_bonus: float = hour_satisfied_bonus
#         self.satisfied_or_nothing: bool = satisfied_or_nothing
#
#     def reward_info(self) -> Optional[Dict[str, Any]]:
#         if self.last_remaining_gap is not None and self.last_cost is not None and self.last_pre_gap is not None:
#             return {
#                 'pre_gap': self.last_pre_gap,
#                 'remaining_gap': self.last_remaining_gap,
#                 'cost': self.last_cost
#             }
#
#     def _compute(self, state_wrapper: Dict[str, State], action: Action, step: Step, demand: StepData, nodes: List[Node],
#                  resource: str, action_space_wrapper: Optional[ActionSpaceWrapper] = None,
#                  pre_gap: float = 0, hour_satisfied: bool = False, **kwargs
#                  ) -> Union[float, Tuple[float, float]]:
#         action_cost = self.compute_action_cost(nodes, resource, action, action_space_wrapper)
#         if self.disable_cost:
#             action_cost = 0
#         remaining_gap = self.compute_remaining_gap(nodes, demand, resource)
#         self.last_cost = action_cost
#         self.last_pre_gap = pre_gap
#         self.last_remaining_gap = remaining_gap
#         if remaining_gap == 0:
#             # remaining gap is 0
#             gap_utility = 1
#         elif remaining_gap <= pre_gap or self.satisfied_or_nothing:
#             # we have more gap than before, or we waited with a gap
#             gap_utility = 0
#         else:
#             # we improved but it is still not solved
#             gap_utility = 0.5
#         cost_utility = normalize_scalar(-action_cost, 0, -self.max_cost[resource], 0, 1)
#         reward = scalarize_rewards(
#             func_type=self.scalarization_function,
#             remaining_gap=gap_utility,
#             cost=cost_utility,
#             initial_gap=self.remaining_gap_max_factor[resource],
#             max_cost=self.max_cost[resource],
#             alpha=self.alpha,
#             normalize_objectives=False,
#             satisfied_nodes=self.get_satisfied_nodes(demand, nodes, resource),
#             n_nodes=len(nodes),
#             only_apply_scalarization=True
#         )
#         if hour_satisfied:
#             reward += self.hour_satisfied_bonus
#         return reward


class GlobalSatisfiedNodes(RewardAbstract):

    def __init__(
            self,
            resources: List[EnvResource],
            env_config,
            run_mode: RunMode,
            disable_cost: bool = False,
            alpha=0.9,
            scalarized: bool = True,
            normalize_objectives: bool = True,
            training_normalization_range: Tuple[int, int] = (0, 1),
            val_eval_normalization_range: Tuple[int, int] = (0, 1),
            scalarization_function: ScalarizationFunction = ScalarizationFunction.Linear,
            action_per_step: int = 1,
            **kwargs
    ):
        super(GlobalSatisfiedNodes, self).__init__(resources=resources, env_config=env_config,
                                                   disable_cost=disable_cost,
                                                   run_mode=run_mode,
                                                   name=RewardFunctionType.GlobalSatisfiedNodes,
                                                   action_per_step=action_per_step, **kwargs)
        self.alpha = alpha
        self.scalarized = scalarized
        self.last_remaining_gap: Optional[float] = None
        self.last_cost: Optional[float] = None
        self.training_normalization_range: Tuple[int, int] = training_normalization_range
        self.val_eval_normalization_range: Tuple[int, int] = val_eval_normalization_range
        self.normalize_objectives: bool = normalize_objectives
        self.scalarization_function: ScalarizationFunction = scalarization_function

    @property
    def normalization_range(self) -> Tuple[int, int]:
        if self.run_mode == RunMode.Train:
            return self.training_normalization_range
        else:
            return self.val_eval_normalization_range

    def reward_info(self) -> Optional[Dict[str, Any]]:
        if self.last_remaining_gap is not None and self.last_cost is not None:
            return {
                'remaining_gap': self.last_remaining_gap,
                'cost': self.last_cost
            }

    def _compute(self, state_wrapper: Dict[str, State], actions: List[Action], step: Step, demand: StepData, nodes: List[Node],
                 resource: str, penalty: Union[int, float] = 0, current_budget=0,
                 action_space_wrapper: Optional[ActionSpaceWrapper] = None, node_groups: Optional[NodeGroups] = None,
                 **kwargs) -> Union[float, Tuple[float, float]]:
        action_cost = self.compute_action_cost(nodes, resource, actions, action_space_wrapper)
        if self.disable_cost:
            action_cost = 0
        satisfied_nodes = self.get_satisfied_nodes(demand, nodes, resource)
        self.last_cost = action_cost
        self.last_remaining_gap = self.compute_remaining_gap(nodes, demand, resource)
        cost_utility = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                        a=self.normalization_range[0], b=self.normalization_range[1])
        nodes_satisfied_utility = normalize_scalar(satisfied_nodes, max_val=len(nodes), min_val=0,
                                                   a=self.normalization_range[0], b=self.normalization_range[1])
        reward = scalarize_rewards(
            func_type=self.scalarization_function,
            remaining_gap=nodes_satisfied_utility,
            cost=cost_utility,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            alpha=self.alpha,
            satisfied_nodes=satisfied_nodes,
            n_nodes=len(nodes),
            only_apply_scalarization=True
        )
        return reward


class GlobalThreeObjectivesReward(RewardAbstract):

    def __init__(
            self,
            resources: List[EnvResource],
            env_config,
            run_mode: RunMode,
            disable_cost: bool = False,
            weights: List[float] = [0.45, 0.1, 0.45],
            normalize_objectives: bool = True,
            training_normalization_range: Tuple[int, int] = (0, 1),
            val_eval_normalization_range: Tuple[int, int] = (0, 1),
            scalarization_function: ScalarizationFunction = ScalarizationFunction.Linear,
            action_per_step: int = 1,
            **kwargs
    ):
        super(GlobalThreeObjectivesReward, self).__init__(resources=resources, env_config=env_config,
                                                          disable_cost=disable_cost,
                                                          run_mode=run_mode,
                                                          name=RewardFunctionType.GlobalThreeObjectives,
                                                          action_per_step=action_per_step, **kwargs)
        self.weights = weights
        self.training_normalization_range: Tuple[int, int] = training_normalization_range
        self.val_eval_normalization_range: Tuple[int, int] = val_eval_normalization_range
        self.normalize_objectives: bool = normalize_objectives
        self.scalarization_function: ScalarizationFunction = scalarization_function

        self.last_remaining_gap: Optional[float] = None
        self.last_cost: Optional[float] = None

    @property
    def normalization_range(self) -> Tuple[int, int]:
        if self.run_mode == RunMode.Train:
            return self.training_normalization_range
        else:
            return self.val_eval_normalization_range

    def reward_info(self) -> Optional[Dict[str, Any]]:
        if self.last_remaining_gap is not None and self.last_cost is not None:
            return {
                'remaining_gap': self.last_remaining_gap,
                'cost': self.last_cost
            }

    def _compute(self, state_wrapper: Dict[str, State], actions: List[Action], step: Step, demand: StepData, nodes: List[Node],
                 resource: str, penalty: Union[int, float] = 0, current_budget=0,
                 action_space_wrapper: Optional[ActionSpaceWrapper] = None, node_groups: Optional[NodeGroups] = None,
                 pre_gap: float = 0,
                 **kwargs) -> Union[float, Tuple[float, float]]:
        action_cost = self.compute_action_cost(nodes, resource, actions, action_space_wrapper)
        res_capacity = 1
        res_units = 1
        for _, res_info in self.resources[0].classes.items():
            res_capacity = res_info['capacity']
            res_units = res_info['allocated'] * len(nodes)
        gap_max = self.remaining_gap_max_factor[resource]
        if self.disable_cost:
            action_cost = 0
        remaining_gap = self.compute_remaining_gap(nodes, demand, resource)
        if self.gap_with_units:
            gap_max = -res_units
        self.last_cost = action_cost
        self.last_remaining_gap = remaining_gap
        if self.normalize_objectives:
            normalized_post_gap = normalize_scalar(remaining_gap, 0,
                                                   gap_max,
                                                   self.normalization_range[0], self.normalization_range[1])
            normalized_cost = normalize_scalar(action_cost, 0,
                                               self.max_cost[resource],
                                               self.normalization_range[0], self.normalization_range[1])
            normalized_pre_gap = normalize_scalar(pre_gap, 0,
                                                  gap_max,
                                                  self.normalization_range[0], self.normalization_range[1])
            values = [normalized_post_gap, normalized_cost, normalized_pre_gap]
        else:
            values = [remaining_gap, action_cost, pre_gap]

        reward = scalarize_rewards(
            func_type=self.scalarization_function,
            remaining_gap=remaining_gap,
            cost=action_cost,
            values=values,
            weights=self.weights,
        )
        return reward


class ResourceUtilizationReward(RewardAbstract):

    def __init__(
            self,
            resources: List[EnvResource],
            env_config,
            run_mode: RunMode,
            disable_cost: bool = False,
            alpha=0.9,
            delta_target: float = 0,
            normalize_objectives: bool = True,
            training_normalization_range: Tuple[int, int] = (0, 1),
            val_eval_normalization_range: Tuple[int, int] = (0, 1),
            scalarization_function: ScalarizationFunction = ScalarizationFunction.Linear,
            training_success_reward: float = 1,
            val_eval_success_reward: float = 1,
            action_per_step: int = 1,
            **kwargs
    ):
        super(ResourceUtilizationReward, self).__init__(resources=resources, env_config=env_config,
                                                        disable_cost=disable_cost,
                                                        run_mode=run_mode,
                                                        name=RewardFunctionType.ResourceUtilization,
                                                        action_per_step=action_per_step, **kwargs)
        self.alpha = alpha
        self.delta_target: float = delta_target
        self.normalize_objectives: bool = normalize_objectives
        self.training_normalization_range: Tuple[int, int] = training_normalization_range
        self.val_eval_normalization_range: Tuple[int, int] = val_eval_normalization_range
        self.scalarization_function: ScalarizationFunction = scalarization_function
        self.training_success_reward: float = training_success_reward
        self.val_eval_success_reward: float = val_eval_success_reward
        self.last_remaining_gap: Optional[float] = None
        self.last_cost: Optional[float] = None
        self.last_surplus: Optional[float] = None
        for res_info in self.resources:
            self.max_cost[res_info.name] *= 2

        self.max_reward = self.val_eval_normalization_range[0]

    @property
    def normalization_range(self) -> Tuple[int, int]:
        if self.run_mode == RunMode.Train:
            return self.training_normalization_range
        else:
            return self.val_eval_normalization_range

    @property
    def success_reward(self) -> float:
        if self.run_mode == RunMode.Train:
            return self.training_success_reward
        else:
            return self.val_eval_success_reward

    def reward_info(self) -> Optional[Dict[str, Any]]:
        if self.last_remaining_gap is not None and self.last_cost is not None and \
                self.last_surplus is not None:
            return {
                'remaining_gap': self.last_remaining_gap,
                'cost': self.last_cost,
                'surplus': self.last_surplus
            }

    def update_reward_info_for_null_reward(self):
        self.last_remaining_gap = 0
        self.last_surplus = 0

    def get_nodes_deltas(self, nodes: List[Node], demand: StepData, resource: str) -> List[float]:
        deltas: List[float] = []
        res_capacity = 1
        if self.gap_with_units:
            for env_res in self.resources:
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

    def _gap_from_deltas(self, deltas: np.array) -> float:
        gap = 0
        for val in deltas:
            if val < self.delta_target:
                gap += (self.delta_target - val)
        if gap > 0:
            gap = gap * -1
        return gap

    def _surplus_from_deltas(self, deltas: np.array) -> float:
        exceeding = 0
        for val in deltas:
            if val > self.delta_target:
                exceeding += (val - self.delta_target)
        return exceeding

    def _compute(self, state_wrapper: Dict[str, State], actions: List[Action], step: Step, demand: StepData, nodes: List[Node],
                 resource: str, penalty: Union[int, float] = 0, current_budget=0,
                 action_space_wrapper: Optional[ActionSpaceWrapper] = None, node_groups: Optional[NodeGroups] = None,
                 **kwargs) -> Union[float, Tuple[float, float]]:
        action_cost = self.compute_action_cost(nodes, resource, actions, action_space_wrapper)
        self.step_action_cost += action_cost
        action_cost = self.step_action_cost
        if self.disable_cost:
            action_cost = 0

        deltas = np.array(self.get_nodes_deltas(nodes, demand, resource))
        distances = l1_distance(deltas, self.delta_target).sum()
        self.last_cost = action_cost
        self.last_remaining_gap = self._gap_from_deltas(deltas)
        self.last_surplus = self._surplus_from_deltas(deltas)
        assert distances == abs(self.last_remaining_gap) + abs(self.last_surplus)
        # normalize reward objectives
        if distances == self.delta_target:
            distances_normalized = self.success_reward
        else:
            if self.normalize_objectives:
                distances_normalized = normalize_scalar(-distances, 0, self.remaining_gap_max_factor[resource] * 2,
                                                        a=self.normalization_range[0], b=self.normalization_range[1])
            else:
                distances_normalized = -distances
        if self.normalize_objectives:
            action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                      a=self.normalization_range[0], b=self.normalization_range[1])
        else:
            action_cost_normalized = -action_cost
        reward = scalarize_rewards(
            func_type=self.scalarization_function,
            remaining_gap=distances_normalized,
            cost=action_cost_normalized,
            initial_gap=self.remaining_gap_max_factor[resource],
            max_cost=self.max_cost[resource],
            alpha=self.alpha,
            normalize_objectives=False,
            satisfied_nodes=self.get_satisfied_nodes(demand, nodes, resource),
            n_nodes=len(nodes),
            only_apply_scalarization=True
        )
        if reward > self.max_reward:
            self.max_reward = reward
        if not self.is_a_null_step(step):
            self.step_action_cost = 0
        return reward


class GapSurplusCostReward(RewardAbstract):

    def __init__(
            self,
            resources: List[EnvResource],
            env_config,
            run_mode: RunMode,
            disable_cost: bool = False,
            weights: Tuple[float, float, float] = (0.45, 0.45, 0.1),
            delta_target: float = 0,
            training_normalization_range: Tuple[int, int] = (0, 1),
            val_eval_normalization_range: Tuple[int, int] = (0, 1),
            action_per_step: int = 1,
            **kwargs
    ):
        super(GapSurplusCostReward, self).__init__(resources=resources, env_config=env_config,
                                                   disable_cost=disable_cost,
                                                   run_mode=run_mode,
                                                   name=RewardFunctionType.GapSurplusCost,
                                                   action_per_step=action_per_step, **kwargs)
        self.weights = weights
        assert abs(1 - sum(self.weights)) < 1e-14
        self.delta_target: float = delta_target
        self.training_normalization_range: Tuple[int, int] = training_normalization_range
        self.val_eval_normalization_range: Tuple[int, int] = val_eval_normalization_range
        self.scalarization_function: ScalarizationFunction = ScalarizationFunction.LinearWeights
        self.last_remaining_gap: Optional[float] = None
        self.last_cost: Optional[float] = None
        self.last_surplus: Optional[float] = None
        for res_info in self.resources:
            self.max_cost[res_info.name] *= 2

    @property
    def normalization_range(self) -> Tuple[int, int]:
        if self.run_mode == RunMode.Train:
            return self.training_normalization_range
        else:
            return self.val_eval_normalization_range

    def reward_info(self) -> Optional[Dict[str, Any]]:
        if self.last_remaining_gap is not None and self.last_cost is not None and \
                self.last_surplus is not None:
            return {
                'remaining_gap': self.last_remaining_gap,
                'surplus': self.last_surplus,
                'cost': self.last_cost,
            }

    def update_reward_info_for_null_reward(self):
        self.last_remaining_gap = 0
        self.last_surplus = 0

    def get_nodes_deltas(self, nodes: List[Node], demand: StepData, resource: str) -> List[float]:
        deltas: List[float] = []
        res_capacity = 1
        if self.gap_with_units:
            for env_res in self.resources:
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

    def _gap_from_deltas(self, deltas: np.array) -> float:
        gap = 0
        for val in deltas:
            if val < self.delta_target:
                gap += (self.delta_target - val)
        if gap > 0:
            gap = gap * -1
        return gap

    def _surplus_from_deltas(self, deltas: np.array) -> float:
        exceeding = 0
        for val in deltas:
            if val > self.delta_target:
                exceeding += (val - self.delta_target)
        return exceeding

    def _compute(self, state_wrapper: Dict[str, State], actions: List[Action], step: Step, demand: StepData, nodes: List[Node],
                 resource: str, penalty: Union[int, float] = 0, current_budget=0,
                 action_space_wrapper: Optional[ActionSpaceWrapper] = None, node_groups: Optional[NodeGroups] = None,
                 **kwargs) -> Union[float, Tuple[float, float]]:
        action_cost = self.compute_action_cost(nodes, resource, actions, action_space_wrapper)
        self.step_action_cost += action_cost
        action_cost = self.step_action_cost
        if self.disable_cost:
            action_cost = 0

        deltas = np.array(self.get_nodes_deltas(nodes, demand, resource))
        self.last_cost = action_cost
        self.last_remaining_gap = self._gap_from_deltas(deltas)
        self.last_surplus = self._surplus_from_deltas(deltas)
        # normalize reward objectives
        gap_normalized = normalize_scalar(self.last_remaining_gap, 0, self.remaining_gap_max_factor[resource],
                                          a=self.normalization_range[0], b=self.normalization_range[1])
        surplus_normalized = normalize_scalar(-self.last_surplus, 0, self.remaining_gap_max_factor[resource],
                                              a=self.normalization_range[0], b=self.normalization_range[1])
        action_cost_normalized = normalize_scalar(-action_cost, 0, -self.max_cost[resource],
                                                  a=self.normalization_range[0], b=self.normalization_range[1])
        reward = scalarize_rewards(
            func_type=self.scalarization_function,
            remaining_gap=gap_normalized,
            cost=action_cost_normalized,
            values=[gap_normalized, surplus_normalized, action_cost_normalized],
            weights=self.weights,
        )
        if not self.is_a_null_step(step):
            self.step_action_cost = 0
        return reward


REWARD_CLASS_MAPPING = {
    RewardFunctionType.GlobalRemainingGap: GlobalRemainingGapReward,
    RewardFunctionType.GlobalSatisfiedNodes: GlobalSatisfiedNodes,
    RewardFunctionType.GlobalThreeObjectives: GlobalThreeObjectivesReward,
    RewardFunctionType.ResourceUtilization: ResourceUtilizationReward,
    RewardFunctionType.GapSurplusCost: GapSurplusCostReward,
}


def get_reward_class(reward_type: RewardFunctionType,
                     env_config,
                     resources: List[EnvResource],
                     n_nodes: int,
                     run_mode: RunMode,
                     disable_cost: bool = False,
                     action_per_step: int = 1,
                     **kwargs):
    if reward_type in REWARD_CLASS_MAPPING:
        rew_class = REWARD_CLASS_MAPPING[reward_type]
        return rew_class(resources=resources,
                         env_config=env_config,
                         n_nodes=n_nodes,
                         run_mode=run_mode,
                         disable_cost=disable_cost,
                         action_per_step=action_per_step,
                         **kwargs)
    else:
        raise RewardFunctionTypeError(reward_type)
